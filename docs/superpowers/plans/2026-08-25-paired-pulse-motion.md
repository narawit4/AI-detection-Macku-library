# Paired-Pulse Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the general two-dimensional jitter generator with a minimal zero-intended-drift vertical paired-pulse engine and a three-setting Motion interface.

**Architecture:** `motion.py` owns the immutable settings snapshot, validation, presets, and a pure stateful `PairedPulseEngine`. `makcu_service.py` retains worker lifecycle and cancellation but schedules at twice the configured pair rate. `settings.py` migrates schema 1 documents to schema 2 defaults without changing unsupported future files, and `ui.py` exposes only Pulse Size, Pulse Rate, and Ramp.

**Tech Stack:** Python 3.12, standard-library dataclasses/threading/json, Tkinter/ttk, `unittest`, external `makcu` runtime package.

**Spec:** `docs/superpowers/specs/2026-08-25-paired-pulse-motion-design.md`

## Global Constraints

- Windows-only; use the supported `makcu` package and relative two-dimensional `move(x, y)` reports.
- Keep all Tk widget and Tk-variable access on the main thread.
- Jitter starts Disabled; Trigger, optional Modifier, Enable, STOP, Test 3s, hotkey, reconnect, disconnect, and shutdown semantics stay intact.
- STOP and all emergency-stop paths signal cancellation immediately and never wait to complete a pair.
- Preserve immutable snapshots under the existing short lock and daemon worker/generation behavior.
- Discard missed pulse intervals; never replay them as a movement backlog.
- Keep `config.json`, backups, logs, EverFall data, generated output, and unsupported future-schema files untouched.
- Keep the fixed `840x620` geometry, persistent runtime dock, visible STOP control, active themes, and Consolas typography.
- Do not add dependencies or run Nuitka.
- The worktree already contains uncommitted rounded-widget and Consolas UI changes. Preserve them. For overlapping `ui.py`, `tests/test_ui.py`, and `tests/test_liquid_widgets.py` commits, stage only this plan's hunks with `git add -p`; never discard or rewrite the existing changes.

---

### Task 1: Paired-Pulse Domain Model and Pure Engine

**Files:**
- Modify: `motion.py`
- Test: `tests/test_motion.py`

**Interfaces:**
- Produces: `RAMP_MODES = ("Instant", "Smooth")`.
- Produces: immutable `MotionSettings(pulse_size_px: float = 2.0, pulse_rate_hz: float = 30.0, ramp_mode: str = "Smooth")`.
- Produces: `MOTION_LIMITS`, `MOTION_DEFAULTS`, and `MOTION_PRESETS` containing only the new keys and the three approved presets.
- Produces: `motion_settings_from_mapping(raw) -> MotionSettings` and `motion_settings_to_mapping(settings) -> dict[str, str]`.
- Produces: `PairedPulseEngine.step(settings: MotionSettings, dt: float, elapsed: float) -> tuple[int, int]` and `reset() -> None`.
- Keeps: `TriggerGate` and its public behavior unchanged.

- [ ] **Step 1: Replace old motion-setting tests with failing tests for the three-field model**

Write literal expectations in `tests/test_motion.py`:

```python
class MotionSettingsTests(unittest.TestCase):
    def test_defaults_match_balanced_paired_pulse(self):
        self.assertEqual(MotionSettings(), MotionSettings(2.0, 30.0, "Smooth"))
        self.assertEqual(
            motion_settings_to_mapping(MotionSettings()),
            {"pulse_size_px": "2", "pulse_rate_hz": "30", "ramp_mode": "Smooth"},
        )

    def test_values_are_clamped_and_invalid_ramp_uses_default(self):
        settings = motion_settings_from_mapping({
            "pulse_size_px": "99",
            "pulse_rate_hz": "-4",
            "ramp_mode": "Unknown",
        })
        self.assertEqual(settings, MotionSettings(8.0, 10.0, "Smooth"))

    def test_presets_are_exact_and_round_trip(self):
        self.assertEqual(tuple(MOTION_PRESETS), ("Soft", "Balanced", "Strong"))
        expected = {
            "Soft": MotionSettings(1.0, 20.0, "Smooth"),
            "Balanced": MotionSettings(2.0, 30.0, "Smooth"),
            "Strong": MotionSettings(4.0, 45.0, "Instant"),
        }
        for name, want in expected.items():
            got = motion_settings_from_mapping(MOTION_PRESETS[name])
            self.assertEqual(got, want)
            self.assertEqual(
                motion_settings_from_mapping(motion_settings_to_mapping(got)),
                want,
            )
```

- [ ] **Step 2: Run the new model tests and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_motion.py -k MotionSettingsTests -v
```

Expected: FAIL because `MotionSettings` still has the old fields and the old preset table.

- [ ] **Step 3: Implement the minimal three-field settings model**

Replace old motion constants, coercion, dataclass fields, and serialization in `motion.py` with:

```python
MOTION_DEFAULTS = {
    "pulse_size_px": "2",
    "pulse_rate_hz": "30",
    "ramp_mode": "Smooth",
}
MOTION_LIMITS = {
    "pulse_size_px": (1.0, 8.0),
    "pulse_rate_hz": (10.0, 60.0),
}
RAMP_MODES = ("Instant", "Smooth")
MOTION_PRESETS = {
    "Soft": {"pulse_size_px": "1", "pulse_rate_hz": "20", "ramp_mode": "Smooth"},
    "Balanced": {"pulse_size_px": "2", "pulse_rate_hz": "30", "ramp_mode": "Smooth"},
    "Strong": {"pulse_size_px": "4", "pulse_rate_hz": "45", "ramp_mode": "Instant"},
}

@dataclass(frozen=True)
class MotionSettings:
    pulse_size_px: float = 2.0
    pulse_rate_hz: float = 30.0
    ramp_mode: str = "Smooth"
```

Keep finite-number validation and compact numeric strings. Clamp numeric values with `MOTION_LIMITS`; select `Smooth` for any ramp value outside `RAMP_MODES`. Remove obsolete boolean, waveform, curve, and random helpers.

- [ ] **Step 4: Run model tests and verify GREEN**

Run the command from Step 2. Expected: all `MotionSettingsTests` pass.

- [ ] **Step 5: Write failing engine tests for pair order, drift, ramp, discard, and reset**

Replace `SmoothMotionEngineTests` with `PairedPulseEngineTests` using these observable sequences:

```python
class PairedPulseEngineTests(unittest.TestCase):
    def test_complete_pairs_alternate_order_and_have_zero_net_motion(self):
        engine = PairedPulseEngine()
        settings = MotionSettings(2.0, 10.0, "Instant")
        reports = [engine.step(settings, 0.05, elapsed) for elapsed in (0.0, 0.05, 0.10, 0.15)]
        self.assertEqual(reports, [(0, -2), (0, 2), (0, 2), (0, -2)])
        self.assertEqual(tuple(map(sum, zip(*reports))), (0, 0))

    def test_many_complete_pairs_never_emit_horizontal_or_net_drift(self):
        engine = PairedPulseEngine()
        settings = MotionSettings(3.0, 20.0, "Instant")
        reports = [engine.step(settings, 0.025, index * 0.025) for index in range(400)]
        self.assertTrue(all(x == 0 for x, _y in reports))
        self.assertEqual(sum(y for _x, y in reports), 0)

    def test_smooth_ramp_accumulates_fraction_without_directional_bias(self):
        engine = PairedPulseEngine()
        settings = MotionSettings(2.0, 30.0, "Smooth")
        reports = [engine.step(settings, 1 / 60, index / 60) for index in range(20)]
        self.assertTrue(all(x == 0 for x, _y in reports))
        self.assertLessEqual(max(abs(y) for _x, y in reports), 2)
        for pair_start in range(0, len(reports), 2):
            self.assertEqual(sum(y for _x, y in reports[pair_start:pair_start + 2]), 0)
        self.assertTrue(any(y for _x, y in reports))

    def test_late_step_discards_missed_half_pulses(self):
        engine = PairedPulseEngine()
        settings = MotionSettings(2.0, 10.0, "Instant")
        self.assertEqual(engine.step(settings, 0.05, 0.0), (0, -2))
        self.assertEqual(engine.step(settings, 0.1, 1.0), (0, 2))

    def test_reset_starts_a_fresh_up_down_pair(self):
        engine = PairedPulseEngine()
        settings = MotionSettings(2.0, 30.0, "Instant")
        engine.step(settings, 1 / 60, 0.0)
        engine.reset()
        self.assertEqual(engine.step(settings, 1 / 60, 0.0), (0, -2))
```

- [ ] **Step 6: Run engine tests and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_motion.py -k PairedPulseEngineTests -v
```

Expected: ERROR importing `PairedPulseEngine` or FAIL because the old engine emits general two-axis motion.

- [ ] **Step 7: Implement `PairedPulseEngine` minimally**

Use explicit state rather than wall-clock callbacks:

```python
@dataclass
class PairedPulseEngine:
    half_pulse_index: int = 0
    magnitude_residual: float = 0.0
    current_pair_size: int = 0
    next_due_elapsed: float = 0.0

    def reset(self) -> None:
        self.half_pulse_index = 0
        self.magnitude_residual = 0.0
        self.current_pair_size = 0
        self.next_due_elapsed = 0.0

    def step(self, settings: MotionSettings, dt: float, elapsed: float) -> tuple[int, int]:
        elapsed = max(0.0, float(elapsed))
        interval = 1.0 / (settings.pulse_rate_hz * 2.0)
        if elapsed + 1e-12 < self.next_due_elapsed:
            return 0, 0
        directions = (-1.0, 1.0, 1.0, -1.0)
        direction = directions[self.half_pulse_index % 4]
        if self.half_pulse_index % 2 == 0:
            ramp = 1.0
            if settings.ramp_mode == "Smooth":
                ramp = min(1.0, elapsed / 0.150)
            magnitude = settings.pulse_size_px * ramp + self.magnitude_residual
            self.current_pair_size = math.trunc(magnitude)
            self.magnitude_residual = magnitude - self.current_pair_size
        report_y = max(-8, min(8, int(direction * self.current_pair_size)))
        self.half_pulse_index += 1
        self.next_due_elapsed = elapsed + interval
        return 0, report_y
```

A zero-sized first Smooth pair still consumes its two scheduled halves; the next
pair starts in the opposite direction as required. Do not add randomness,
horizontal motion, compensation-on-stop, or a missed-interval loop.

- [ ] **Step 8: Run all motion tests and verify GREEN**

Run:

```powershell
python -m unittest discover -s tests -p test_motion.py -v
```

Expected: all settings, engine, and unchanged TriggerGate tests pass.

- [ ] **Step 9: Commit Task 1**

```powershell
git add -- motion.py tests/test_motion.py
git commit -m "feat: replace motion engine with paired pulses"
```

---

### Task 2: Schema 2 Migration and Atomic Persistence

**Files:**
- Modify: `settings.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Consumes: `MotionSettings`, `motion_settings_from_mapping`, and `motion_settings_to_mapping` from Task 1.
- Produces: `SCHEMA_VERSION = 2`.
- Produces: schema 1 load behavior that preserves trigger/modifier/hotkey/theme and uses `MotionSettings()` plus `selected_preset="Custom"`.
- Keeps: `ConfigStore.save(config)` atomic temp/flush/fsync/backup/replace behavior and future-schema save protection.

- [ ] **Step 1: Write failing schema migration and round-trip tests**

Update old motion field usage and add:

```python
def test_schema_one_preserves_app_choices_but_migrates_motion_to_defaults(self):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "config.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "motion": {"motion_strength_pps": "123"},
            "trigger": "Right",
            "modifier": "Mouse4",
            "hotkey_vk": 65,
            "hotkey_name": "A",
            "selected_preset": "Strong Shake",
            "theme": "dark",
        }), encoding="utf-8")
        outcome = ConfigStore(path).load()
        self.assertEqual(outcome.config.motion, MotionSettings())
        self.assertEqual(outcome.config.trigger, "Right")
        self.assertEqual(outcome.config.modifier, "Mouse4")
        self.assertEqual(outcome.config.hotkey_vk, 65)
        self.assertEqual(outcome.config.hotkey_name, "A")
        self.assertEqual(outcome.config.selected_preset, "Custom")
        self.assertEqual(outcome.config.theme, "dark")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema_version"], 1)

def test_schema_two_round_trip_saves_only_paired_pulse_motion(self):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "config.json"
        store = ConfigStore(path)
        config = AppConfig(
            motion=MotionSettings(4.0, 45.0, "Instant"),
            selected_preset="Strong",
        )
        store.save(config)
        document = json.loads(path.read_text(encoding="utf-8"))
        restored = store.load().config
    self.assertEqual(document["schema_version"], 2)
    self.assertEqual(document["motion"], {
        "pulse_size_px": "4", "pulse_rate_hz": "45", "ramp_mode": "Instant",
    })
    self.assertEqual(restored, config)
```

Retain tests that corrupt JSON uses defaults, malformed schema 2 values are safe, second save keeps a backup, and schema 3 is neither overwritten nor save-enabled.

- [ ] **Step 2: Run settings tests and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_settings.py -v
```

Expected: FAIL because `SCHEMA_VERSION` is 1 and schema 1 currently parses old motion instead of migrating it.

- [ ] **Step 3: Implement schema 2 branching**

Set `SCHEMA_VERSION = 2`. In `load()`, validate the common non-motion fields as today, but choose motion and preset by schema:

```python
if schema == 1:
    motion = MotionSettings()
    selected_preset = "Custom"
else:
    motion_raw = document.get("motion")
    motion = motion_settings_from_mapping(
        motion_raw if isinstance(motion_raw, Mapping) else None
    )
    selected_preset = document.get("selected_preset", "Custom")
    if selected_preset not in {"Custom", *MOTION_PRESETS}:
        selected_preset = "Custom"
```

Import `MOTION_PRESETS`. Do not save during `load()`. Keep `schema > 2` behavior and the atomic `save()` body intact apart from the new serialized motion shape.

- [ ] **Step 4: Run settings tests and verify GREEN**

Run the command from Step 2. Expected: all settings tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- settings.py tests/test_settings.py
git commit -m "feat: migrate configuration to paired-pulse schema"
```

---

### Task 3: Movement Worker Pulse Scheduling and Immediate Cancellation

**Files:**
- Modify: `makcu_service.py`
- Test: `tests/test_makcu_service.py`

**Interfaces:**
- Consumes: `PairedPulseEngine.step(settings, dt, elapsed)` and `MotionSettings.pulse_rate_hz` from Task 1.
- Produces: default `engine_factory=PairedPulseEngine` and worker wait interval `1 / (pulse_rate_hz * 2)`.
- Keeps: `start_motion`, `stop_motion`, `join_motion`, generation checks, duration completion, and service event contracts unchanged.

- [ ] **Step 1: Update fakes and write failing worker tests**

Replace test settings carrying `update_rate_hz` with `MotionSettings(..., pulse_rate_hz=20)`. Add a recording fake engine and assert the worker's output remains vertical and cancellation prevents a second report:

```python
def test_paired_pulse_worker_sends_vertical_reports_and_stop_prevents_next_half(self):
    self.assertTrue(self.service.start_motion(lambda: MotionSettings(2, 20, "Instant")))
    self.assertTrue(wait_until(lambda: len(self.controller.moves) >= 1))
    self.service.stop_motion("manual")
    moves_after_stop = list(self.controller.moves)
    time.sleep(0.06)
    self.assertEqual(self.controller.moves, moves_after_stop)
    self.assertTrue(all(x == 0 and abs(y) <= 2 for x, y in moves_after_stop))
```

Keep the existing lock-based assertion that once `stop_motion()` returns, no report for that generation can be sent.

- [ ] **Step 2: Run Makcu movement tests and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_makcu_service.py -k MakcuMovementTests -v
```

Expected: FAIL or ERROR because the service still imports the old engine and reads `update_rate_hz`.

- [ ] **Step 3: Switch the service to paired-pulse timing**

Import `PairedPulseEngine`, use it as the constructor default, and replace the worker interval calculation with:

```python
settings_rate = float(settings.pulse_rate_hz)
interval = 1.0 / (max(10.0, min(60.0, settings_rate)) * 2.0)
stop_event.wait(max(0.0, interval - (time.perf_counter() - tick_started)))
```

Do not change the stop-event check immediately before `controller.move()`. Do not complete or compensate a half pair during teardown.

- [ ] **Step 4: Run service tests and verify GREEN**

Run:

```powershell
python -m unittest discover -s tests -p test_makcu_service.py -v
```

Expected: all connection, movement, stale-generation, error, and cancellation tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- makcu_service.py tests/test_makcu_service.py
git commit -m "feat: schedule paired pulses in Makcu worker"
```

---

### Task 4: Minimal Motion UI and Two-Destination Navigation

**Files:**
- Modify: `ui.py`
- Modify: `tests/test_ui.py`
- Modify if a navigation assumption needs general coverage: `tests/test_liquid_widgets.py`

**Interfaces:**
- Consumes: `MOTION_LIMITS`, `MOTION_PRESETS`, `RAMP_MODES`, `MotionSettings`, and mapping helpers from Task 1.
- Produces: `Control` and `Motion` navigation destinations only.
- Produces: `pulse_size_px_var`, `pulse_rate_hz_var`, `ramp_mode_var`, exact entries/sliders, `ramp_mode_combo`, and paired-pulse summary text.
- Removes: Advanced page/canvas/scrollbar, old motion variables, waveform/curve selectors, and old public aliases that refer only to removed controls.
- Keeps: runtime dock, STOP, footer, device controls, trigger/modifier, preset, hotkey, Test 3s, theme, fixed geometry, state preservation, and main-thread UI access.

- [ ] **Step 1: Replace old layout assertions with failing minimal-UI tests**

Update tests that expect three pages, Advanced scrolling, old numeric keys, or old summaries. Add literal contract tests:

```python
def test_navigation_contains_only_control_and_motion(self):
    self.assertEqual(self.app.nav.labels, ("Control", "Motion"))
    self.assertEqual(self.app.pages, (self.app.control_page, self.app.motion_page))
    self.assertFalse(hasattr(self.app, "advanced_page"))
    self.assertFalse(hasattr(self.app, "advanced_canvas"))

def test_motion_page_exposes_only_paired_pulse_controls(self):
    self.assertEqual(
        set(self.app.motion_vars),
        {"pulse_size_px", "pulse_rate_hz", "ramp_mode"},
    )
    self.assertEqual(self.app.ramp_mode_combo.cget("values"), ("Instant", "Smooth"))
    self.assertEqual(self.app.preset_values, ("Custom", "Soft", "Balanced", "Strong"))

def test_motion_summary_describes_paired_pulse_snapshot(self):
    self.app._replace_motion_snapshot(MotionSettings(2.0, 30.0, "Smooth"))
    self.assertEqual(
        self.app.motion_summary_var.get(),
        "2 px paired pulse at 30 Hz | Smooth",
    )

def test_pulse_edits_update_immutable_snapshot(self):
    self.app.pulse_size_px_var.set("4")
    self.app.pulse_rate_hz_var.set("45")
    self.app.ramp_mode_var.set("Instant")
    self.app._motion_changed("pulse_size_px")
    self.assertEqual(self.app.get_motion_settings(), MotionSettings(4.0, 45.0, "Instant"))
```

Delete tests whose only contract is ownership or scrolling of the removed Advanced page. Retain geometry, STOP visibility, page switching, theme, input validation, Test 3s, Trigger/Modifier, hotkey, and shutdown tests with new field names.

- [ ] **Step 2: Run UI tests and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_ui.py -v
```

Expected: FAIL because the UI still builds Advanced and the old motion controls.

- [ ] **Step 3: Reduce imports, variables, summary, pages, and navigation**

In `ui.py`:

```python
def _motion_summary_text(settings: MotionSettings) -> str:
    return (
        f"{_display_value(settings.pulse_size_px)} px paired pulse at "
        f"{_display_value(settings.pulse_rate_hz)} Hz | {settings.ramp_mode}"
    )
```

Import `RAMP_MODES`; remove `JITTER_WAVEFORMS` and `MOTION_CURVES`. Build `LiquidNavigation(labels=("Control", "Motion"), ...)`, create only two pages, remove `_build_advanced_workspace()` and `_build_advanced_card()` calls, and remove Advanced wheel bindings/callback cleanup that has no remaining owner. Keep `select_page()` generic over `self.pages`.

- [ ] **Step 4: Build the minimal Motion card**

Use `_numeric_control` for:

```python
controls = (
    ("Pulse Size", "pulse_size_px", 1, 8, 1),
    ("Pulse Rate", "pulse_rate_hz", 10, 60, 1),
)
```

Add one readonly `ramp_mode_combo` bound to `self.motion_vars["ramp_mode"]` with `values=RAMP_MODES`. Keep the live snapshot card and preset combo on the Control page. Ensure `_apply_combobox_popup_palette()` includes only live comboboxes. Mapping-driven variable creation, invalid numeric validation, preset application, immutable snapshot replacement, and delayed atomic save remain the single data path.

- [ ] **Step 5: Remove obsolete Advanced-only methods and aliases**

Delete `_build_advanced_workspace`, `_build_advanced_card`, `_refresh_scrollregion`, `_resize_content_window`, Advanced-only wheel enter/leave routing, obsolete widget attributes, and cancellation paths that only reference the removed canvas. Do not remove footer/runtime ownership or STOP placement. Search production UI for every removed field:

```powershell
rg -n "motion_strength|angle_deg|horizontal_jitter|vertical_jitter|smoothness|update_rate|ramp_up|max_step|acceleration|deceleration|jitter_waveform|motion_curve|advanced_(page|canvas|scrollbar|content)" ui.py
```

Expected: no matches, except prose comments that are still accurate; remove inaccurate comments.

- [ ] **Step 6: Run UI and liquid-widget tests and verify GREEN**

Run:

```powershell
python -m unittest discover -s tests -p test_ui.py -v
python -m unittest discover -s tests -p test_liquid_widgets.py -v
```

Expected: all UI tests pass, navigation works with two labels in both orientations, fixed geometry remains `840x620`, and STOP stays visible on both pages.

- [ ] **Step 7: Commit Task 4 without staging pre-existing UI hunks**

Inspect `git diff` and interactively stage only paired-pulse UI/test hunks:

```powershell
git add -p -- ui.py tests/test_ui.py tests/test_liquid_widgets.py
git diff --cached --check
git commit -m "feat: simplify UI for paired-pulse motion"
```

Do not stage unrelated rounded-widget or Consolas changes already present before this plan.

---

### Task 5: Cross-Layer Verification and Documentation Alignment

**Files:**
- Modify: `README.md`
- Test: `tests/test_entrypoints.py`
- Test: all files under `tests/`

**Interfaces:**
- Consumes: completed Tasks 1-4.
- Produces: user-facing documentation that describes paired pulses and the reduced interface without claiming guaranteed stabilization.
- Keeps: source launcher, dependency policy, packaging policy, and hardware-free CI behavior.

- [ ] **Step 1: Write a failing documentation behavior test**

Add a narrow source-tree documentation test in `tests/test_entrypoints.py`:

```python
def test_readme_describes_paired_pulse_controls_without_obsolete_motion_terms(self):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    self.assertIn("Pulse Size", text)
    self.assertIn("Pulse Rate", text)
    self.assertIn("up/down", text)
    for obsolete in ("Motion Angle", "Randomness", "Waveform", "Acceleration"):
        self.assertNotIn(obsolete, text)
```

- [ ] **Step 2: Run the documentation test and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_entrypoints.py -k paired_pulse_controls -v
```

Expected: FAIL because README still documents the old general jitter controls.

- [ ] **Step 3: Update README behavior and limitations**

Describe vertical paired pulses, the three controls/presets, unchanged gating and emergency-stop behavior, Test 3s, and the statement: “Complete pulse pairs have zero intended displacement; results vary with the receiving application's input processing.” Remove obsolete field/preset/Advanced instructions. Do not claim guaranteed aim stabilization.

- [ ] **Step 4: Run complete verification**

Run exactly:

```powershell
python -m py_compile main.py ui.py motion.py makcu_service.py hotkeys.py settings.py liquid_widgets.py
python -m unittest discover -s tests -v
python -c "import makcu"
git diff --check
```

Expected: compile exits 0, every test passes with zero failures/errors, Makcu imports successfully, and diff check reports no whitespace errors. Do not run Nuitka.

- [ ] **Step 5: Record hardware verification as pending unless a device is connected**

With a connected Makcu, manually verify connection, Trigger plus Modifier, exact up/down direction, Soft/Balanced/Strong rate and size, Test 3s, STOP between half-pulses, hotkey disable, disconnect, reconnect, and shutdown. Without hardware, report these checks explicitly as not performed; do not infer success from mocks.

- [ ] **Step 6: Commit Task 5**

```powershell
git add -- README.md tests/test_entrypoints.py
git commit -m "docs: describe paired-pulse controls"
```
