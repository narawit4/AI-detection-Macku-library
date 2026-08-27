# Conservative AI Tracker, Response Curve, and Adaptive Cadence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Binding schema supersession:** Extend the existing schema 5 document with
> the response curve. Schema 6 is unsupported future data, saving stays
> disabled after it is loaded, and its source bytes remain unchanged. This
> ruling supersedes every schema-6 instruction in earlier plan revisions.

**Goal:** Prevent close or overlapping detections from stealing the active AI target, and replace burst-per-inference mouse movement with a configurable five-point, display-adaptive time servo.

**Architecture:** A pure `ai_tracking.py` state machine scores full base boxes using predicted position, IoU, and area continuity, publishing no target when identity is ambiguous. `ai_targeting.py` owns validated curve settings, monotone interpolation, and a `dt`-based movement servo; `display_timing.py` derives immutable capture/servo cadence from the primary Windows display. `AiService`, Makcu composition, schema 5, and the scrollable Tk Motion page consume those narrow interfaces without weakening generation or cancellation barriers.

**Tech Stack:** Python 3.12, dataclasses, ctypes/Win32, Tkinter Canvas/ttk, NumPy/DXCam, ONNX Runtime DirectML, Makcu, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-27-conservative-ai-tracker-response-curve-design.md`

## Global Constraints

- Windows-only; query the current primary display because AI capture remains DXCam output index 0.
- Keep the fixed centered 320-by-320 capture, bundled `models/all_games_320.onnx`, DirectML-first runtime, and at most two detector calls per captured frame.
- Add no dependency, model, training, profile, tray behavior, alternate runtime, Pillow, OpenCV, Torch, or Ultralytics.
- Ambiguity publishes current Overlay boxes with `target=None` and `selected_index=None`; it never guesses or holds a stale movement coordinate.
- Jitter continues while AI is ambiguous, missing, pending replacement, or otherwise contributes `(0, 0)`.
- The confirmed original requires two clear observations after ambiguity; a replacement requires three stable observations within the inclusive 18-pixel boundary.
- Tracker predictive hold and movement freshness both expire at elapsed time `>= 0.150` seconds.
- Valid display rates are finite 24..500 Hz inclusive; capture is capped at 240 FPS, servo is `clamp(2 * display_hz, 120, 480)`, and fallback is 120/240.
- Response X positions are fixed at 0/25/50/75/100%; Y values are finite, 0..1, monotonic, exactly five long, and begin at zero.
- Keep Confidence, Aim Strength, Smoothing, and Max Step; persist only the response curve in schema 5, never cadence/tracker/runtime state.
- Tk widgets and variables stay on the main thread; workers receive immutable snapshots and injected runtime values.
- STOP, disconnect, hotkey disable, source change, Trigger/Modifier release, error, restart, and shutdown retain immediate signaling and the existing Makcu move barrier.
- Preserve fractional accumulation, per-report clamping, no overshoot, and discard obsolete/excess movement instead of queuing it.
- Do not run Nuitka unless the user explicitly requests a package build.

## File Structure

- Create `display_timing.py`: Win32 refresh query plus pure `RuntimeCadence` policy.
- Create `ai_tracking.py`: immutable base-box tracker, scoring, ambiguity, recovery, and replacement confirmation.
- Modify `ai_targeting.py`: response-curve contract/interpolation and time-based `AimMovementEngine`.
- Verify `ai_capture.py`: retain its existing injectable `target_fps` contract; no capture geometry change is required.
- Modify `ai_service.py`: injectable capture FPS and per-generation conservative tracker integration.
- Modify `combined_motion.py`: injected AI servo cadence and default cadence-aware aim engine.
- Modify `makcu_service.py`: pass validated AI servo cadence into its default combined-engine factory.
- Modify `ui.py`: runtime cadence injection/status and the scrollable response-curve editor.
- Modify `settings.py`: schema 5 response-curve extension and persistence.
- Modify `README.md` and `AGENTS.md`: behavior, schema, layout, and verification inventory.
- Create `tests/test_display_timing.py` and `tests/test_ai_tracking.py`; extend the existing targeting, capture, service, composition, Makcu, settings, entrypoint, and UI tests.

---

### Task 1: Primary-display cadence policy

**Files:**
- Create: `display_timing.py`
- Create: `tests/test_display_timing.py`

**Interfaces:**
- Consumes: injectable Win32 `user32.EnumDisplaySettingsW` boundary.
- Produces: `RuntimeCadence(display_hz: int | None, capture_fps: int, servo_hz: int)`, `cadence_from_refresh(raw: Any) -> RuntimeCadence`, and `detect_runtime_cadence(user32: Any | None = None) -> RuntimeCadence`.

- [ ] **Step 1: Write failing pure cadence tests**

```python
from display_timing import RuntimeCadence, cadence_from_refresh


def test_valid_refresh_drives_capture_and_double_rate_servo(self):
    self.assertEqual(
        cadence_from_refresh(144),
        RuntimeCadence(display_hz=144, capture_fps=144, servo_hz=288),
    )

def test_caps_and_fallback_are_exact(self):
    self.assertEqual(cadence_from_refresh(360), RuntimeCadence(360, 240, 480))
    for raw in (None, True, float("nan"), 23.99, 500.01, "bad"):
        with self.subTest(raw=raw):
            self.assertEqual(cadence_from_refresh(raw), RuntimeCadence(None, 120, 240))

def test_valid_boundaries_and_rounding(self):
    self.assertEqual(cadence_from_refresh(24), RuntimeCadence(24, 24, 120))
    self.assertEqual(cadence_from_refresh(500), RuntimeCadence(500, 240, 480))
    self.assertEqual(cadence_from_refresh(143.6), RuntimeCadence(144, 144, 288))
```

Add injected Win32 tests whose fake `EnumDisplaySettingsW` writes
`dmDisplayFrequency=165`, returns false, or raises `OSError`; assert success is
`RuntimeCadence(165, 165, 330)` and both failures return the fallback without
raising.

- [ ] **Step 2: Run the cadence tests and confirm RED**

Run: `python -m unittest tests.test_display_timing -v`

Expected: import failure because `display_timing.py` does not exist.

- [ ] **Step 3: Implement the immutable cadence and Win32 adapter**

```python
@dataclass(frozen=True)
class RuntimeCadence:
    display_hz: int | None
    capture_fps: int
    servo_hz: int


FALLBACK_CADENCE = RuntimeCadence(None, 120, 240)


def cadence_from_refresh(raw: Any) -> RuntimeCadence:
    try:
        if isinstance(raw, bool):
            raise TypeError
        value = float(raw)
        if not math.isfinite(value) or not 24.0 <= value <= 500.0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        return FALLBACK_CADENCE
    display_hz = int(round(value))
    return RuntimeCadence(
        display_hz,
        min(display_hz, 240),
        max(120, min(480, display_hz * 2)),
    )
```

Define a correctly sized `DEVMODEW` ctypes structure, set `dmSize`, and call
`EnumDisplaySettingsW(None, ENUM_CURRENT_SETTINGS, byref(mode))`. Return the
fallback when the call fails or an exception is raised; log the exception with
`logging.exception` but do not expose it as a UI/runtime error. Keep the API
object injectable.

Use the complete display-mode layout so `dmDisplayFrequency` has the native
Windows offset:

```python
class _POINTL(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

class _PRINTER_FIELDS(ctypes.Structure):
    _fields_ = [
        ("dmOrientation", wintypes.SHORT), ("dmPaperSize", wintypes.SHORT),
        ("dmPaperLength", wintypes.SHORT), ("dmPaperWidth", wintypes.SHORT),
        ("dmScale", wintypes.SHORT), ("dmCopies", wintypes.SHORT),
        ("dmDefaultSource", wintypes.SHORT), ("dmPrintQuality", wintypes.SHORT),
    ]

class _DISPLAY_FIELDS(ctypes.Structure):
    _fields_ = [
        ("dmPosition", _POINTL),
        ("dmDisplayOrientation", wintypes.DWORD),
        ("dmDisplayFixedOutput", wintypes.DWORD),
    ]

class _MODE_FIELDS(ctypes.Union):
    _fields_ = [("printer", _PRINTER_FIELDS), ("display", _DISPLAY_FIELDS)]

class _DISPLAY_FLAGS(ctypes.Union):
    _fields_ = [("dmDisplayFlags", wintypes.DWORD), ("dmNup", wintypes.DWORD)]

class _DEVMODEW(ctypes.Structure):
    _anonymous_ = ("mode", "flags")
    _fields_ = [
        ("dmDeviceName", wintypes.WCHAR * 32),
        ("dmSpecVersion", wintypes.WORD), ("dmDriverVersion", wintypes.WORD),
        ("dmSize", wintypes.WORD), ("dmDriverExtra", wintypes.WORD),
        ("dmFields", wintypes.DWORD), ("mode", _MODE_FIELDS),
        ("dmColor", wintypes.SHORT), ("dmDuplex", wintypes.SHORT),
        ("dmYResolution", wintypes.SHORT), ("dmTTOption", wintypes.SHORT),
        ("dmCollate", wintypes.SHORT), ("dmFormName", wintypes.WCHAR * 32),
        ("dmLogPixels", wintypes.WORD), ("dmBitsPerPel", wintypes.DWORD),
        ("dmPelsWidth", wintypes.DWORD), ("dmPelsHeight", wintypes.DWORD),
        ("flags", _DISPLAY_FLAGS), ("dmDisplayFrequency", wintypes.DWORD),
        ("dmICMMethod", wintypes.DWORD), ("dmICMIntent", wintypes.DWORD),
        ("dmMediaType", wintypes.DWORD), ("dmDitherType", wintypes.DWORD),
        ("dmReserved1", wintypes.DWORD), ("dmReserved2", wintypes.DWORD),
        ("dmPanningWidth", wintypes.DWORD), ("dmPanningHeight", wintypes.DWORD),
    ]
```

- [ ] **Step 4: Run focused and baseline-adjacent tests**

Run: `python -m unittest tests.test_display_timing tests.test_entrypoints -v`

Expected: PASS; importing the new module must not alter the lazy AI self-check.

- [ ] **Step 5: Commit the cadence policy**

```powershell
git add display_timing.py tests/test_display_timing.py
git commit -m "feat: derive AI cadence from the primary display"
```

---

### Task 2: Five-point curve settings and monotone interpolation

**Files:**
- Modify: `ai_targeting.py`
- Modify: `tests/test_ai_targeting.py`

**Interfaces:**
- Consumes: fixed X positions `(0.0, 0.25, 0.5, 0.75, 1.0)`.
- Produces: `DEFAULT_RESPONSE_CURVE`, `validated_response_curve(raw: Any) -> tuple[float, float, float, float, float]`, `response_curve_value(curve: tuple[float, ...], normalized_distance: float) -> float`, and `AimSettings.response_curve`.

- [ ] **Step 1: Write failing validation and interpolation tests**

```python
def test_response_curve_defaults_and_round_trips(self):
    settings = aim_settings_from_mapping({
        "response_curve": ["0", "0.1", "0.3", "0.7", "0.9"],
    })
    self.assertEqual(settings.response_curve, (0.0, 0.1, 0.3, 0.7, 0.9))
    self.assertEqual(
        aim_settings_to_mapping(settings)["response_curve"],
        ["0", "0.1", "0.3", "0.7", "0.9"],
    )

def test_malformed_curve_uses_complete_default(self):
    invalid = (
        None, [0, .1], [0, .4, .3, .8, 1],
        [.1, .2, .3, .4, .5], [0, .2, .3, .4, float("nan")],
    )
    for raw in invalid:
        with self.subTest(raw=raw):
            self.assertEqual(validated_response_curve(raw), DEFAULT_RESPONSE_CURVE)

def test_monotone_curve_hits_points_and_never_overshoots(self):
    curve = (0.0, 0.12, 0.35, 0.68, 1.0)
    for x, y in zip(RESPONSE_CURVE_X, curve):
        self.assertAlmostEqual(response_curve_value(curve, x), y)
    samples = [response_curve_value(curve, index / 100) for index in range(101)]
    self.assertEqual(samples, sorted(samples))
    self.assertEqual(response_curve_value(curve, -1), 0.0)
    self.assertEqual(response_curve_value(curve, 2), 1.0)
```

Also cover boolean elements, infinities, values outside 0..1, a valid flat
segment, and frozen `AimSettings` immutability.

- [ ] **Step 2: Run the curve tests and confirm RED**

Run: `python -m unittest tests.test_ai_targeting.AimSettingsTests -v`

Expected: import/attribute failures for the curve symbols and fifth settings field.

- [ ] **Step 3: Add curve validation without breaking positional settings callers**

Append the field so the existing four positional arguments remain valid:

```python
RESPONSE_CURVE_X = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_RESPONSE_CURVE = (0.0, 0.12, 0.35, 0.68, 1.0)

@dataclass(frozen=True)
class AimSettings:
    confidence: float = 0.35
    aim_strength: float = 0.35
    smoothing: float = 0.65
    max_step: int = 20
    response_curve: tuple[float, float, float, float, float] = DEFAULT_RESPONSE_CURVE
```

Reject the entire input unless it is an ordered, non-string sequence of exactly
five finite non-boolean numbers or numeric strings, begins exactly at zero,
stays within 0..1, and is non-decreasing. Mappings and unordered iterables use
the complete default. Serialize valid data as a five-element list of compact
strings while keeping the four scalar mapping values unchanged. Update the
serializer return annotation to `dict[str, str | list[str]]`; consumers that
need only scalar Tk variables must select `_AI_CONTROL_SPECS` keys rather than
iterating the whole mapping.

- [ ] **Step 4: Implement monotone cubic Hermite evaluation**

Compute secants between the fixed X positions, derive Fritsch-Carlson monotone
tangents, zero tangents around flat secants, and evaluate the containing
segment with cubic Hermite basis functions. Clamp both input X and final Y.
Do not import SciPy or NumPy for scalar curve evaluation.

- [ ] **Step 5: Run the full targeting tests**

Run: `python -m unittest tests.test_ai_targeting -v`

Expected: PASS, including existing confidence/selection/movement tests.

- [ ] **Step 6: Commit the curve contract**

```powershell
git add ai_targeting.py tests/test_ai_targeting.py
git commit -m "feat: add validated AI response curves"
```

---

### Task 3: Schema 5 response-curve persistence

**Files:**
- Modify: `settings.py`
- Modify: `tests/test_settings.py`

**Interfaces:**
- Consumes: `AimSettings.response_curve` and its mapping helpers from Task 2.
- Produces: `SCHEMA_VERSION = 5`, safe schema 1-4 migration, an atomic schema 5 round trip, and schema-6 future-data protection.

- [ ] **Step 1: Write failing schema 5 tests**

```python
def test_schema_five_round_trips_response_curve(self):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        config = AppConfig(ai=AimSettings(
            0.5, 0.6, 0.7, 30, (0.0, 0.1, 0.4, 0.8, 0.9)
        ))
        store = ConfigStore(path)
        store.save(config)
        document = json.loads(path.read_text(encoding="utf-8"))
        restored = store.load().config
    self.assertEqual(document["schema_version"], 5)
    self.assertEqual(document["ai"]["response_curve"], ["0", "0.1", "0.4", "0.8", "0.9"])
    self.assertEqual(restored, config)

def test_schema_five_receives_default_curve_without_rewrite(self):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        original = json.dumps({
            "schema_version": 5,
            "ai": {"aim_strength": "0.8"},
        })
        path.write_text(original, encoding="utf-8")
        restored = ConfigStore(path).load().config
        after = path.read_text(encoding="utf-8")
    self.assertEqual(restored.ai.response_curve, DEFAULT_RESPONSE_CURVE)
    self.assertEqual(after, original)
```

Add cases for malformed schema 5 curves, schemas 1-4 defaulting the curve,
unsupported schema 6 disabling save without overwrite, and second-save backup
retaining the complete preceding schema 5 document.

- [ ] **Step 2: Run settings tests and confirm RED**

Run: `python -m unittest tests.test_settings -v`

Expected: response-curve assertions fail because schema 5 does not yet
serialize the curve.

- [ ] **Step 3: Extend schema 5 minimally**

Keep `SCHEMA_VERSION = 5`. Keep schema 1/2 AI defaults, let schemas 3/4 retain
their scalar AI settings while ignoring any response curve, and let schema 5
pass its mapping through `aim_settings_from_mapping`; missing curve data
naturally uses Task 2's default. Treat schema 6 as unsupported future data and
keep save prohibition, byte preservation, temp cleanup, flush, `fsync`, backup,
and `os.replace` unchanged.

- [ ] **Step 4: Run settings and UI config tests**

Run: `python -m unittest tests.test_settings tests.test_ui.JitterLayoutTests.test_save_config_persists_only_settings_not_runtime_state -v`

Expected: PASS; no cadence, target, or runtime key appears in JSON.

- [ ] **Step 5: Commit schema 5 persistence**

```powershell
git add settings.py tests/test_settings.py
git commit -m "feat: persist response curves in schema five"
```

---

### Task 4: Time-based AI movement servo

**Files:**
- Modify: `ai_targeting.py`
- Modify: `combined_motion.py`
- Modify: `tests/test_ai_targeting.py`
- Modify: `tests/test_combined_motion.py`

**Interfaces:**
- Consumes: `response_curve_value`, immutable `AimSettings`, `TargetSnapshot`, measured monotonic `now`, and injected nominal servo Hz.
- Produces: `AimMovementEngine(nominal_hz: float = 240.0)` that emits repeated microsteps and `CombinedMotionEngine(..., ai_poll_hz: float = 240.0)`.

- [ ] **Step 1: Replace burst assumptions with failing servo tests**

```python
def test_one_fresh_snapshot_produces_multiple_microsteps(self):
    engine = AimMovementEngine(nominal_hz=240)
    target = TargetSnapshot(1, 10.0, "head", 210.0, 160.0)
    reports = [
        engine.step(target, AimSettings(smoothing=0.0), 10.0 + index / 240)
        for index in range(8)
    ]
    self.assertGreater(sum(report[0] != 0 for report in reports), 1)
    self.assertTrue(all(report[1] == 0 for report in reports))

def test_elapsed_time_not_tick_count_controls_total_displacement(self):
    totals = []
    for hz in (120, 288, 480):
        engine = AimMovementEngine(nominal_hz=hz)
        target = TargetSnapshot(1, 20.0, "head", 220.0, 160.0)
        reports = [engine.step(target, AimSettings(smoothing=0.0), 20 + i / hz)
                   for i in range(int(hz * 0.1))]
        totals.append(sum(x for x, _ in reports))
    self.assertLessEqual(max(totals) - min(totals), 2)
```

Add exact tests for the first nominal interval, fresh-sequence error
replacement, Smoothing 0 and 0.95, 21,600 px/s² acceleration, Max Step,
fractional carry, dead zone, 150 ms exact staleness, `None` reset, backward or
large clock jumps, per-axis no-overshoot, and no obsolete excess after a new
snapshot.

- [ ] **Step 2: Run movement tests and confirm RED**

Run: `python -m unittest tests.test_ai_targeting.AimMovementEngineTests tests.test_combined_motion -v`

Expected: repeated sequences currently return zero and cadence constructor arguments are rejected.

- [ ] **Step 3: Implement servo state and exact formulas**

`reset()` clears last sequence, estimated remaining error, velocity, fractions,
previous tick, and target capture time. The first eligible call uses
`1 / nominal_hz`; later calls clamp measured `dt` to 0..0.1 seconds.

For remaining vector `(rx, ry)`:

```python
radius = math.hypot(rx, ry)
normalized = min(1.0, radius / math.hypot(160.0, 160.0))
curve_distance = response_curve_value(settings.response_curve, normalized) * math.hypot(160.0, 160.0)
reference_step = min(float(settings.max_step), curve_distance * settings.aim_strength)
desired_speed = reference_step * 60.0
desired_vx = desired_speed * rx / radius
desired_vy = desired_speed * ry / radius
```

For positive Smoothing use
`tau = 0.200 * (smoothing / 0.95) ** 2` and
`alpha = 1 - exp(-dt / tau)`; Smoothing 0 uses alpha 1. Clamp velocity change
to `21600 * dt` per vector magnitude, integrate `velocity * dt`, add fractions,
truncate to integers, clamp each report to ±Max Step and to the sign/magnitude
of remaining error, then subtract only emitted integers. Stale/None/dead-zone
input calls `reset()` and returns zero.

- [ ] **Step 4: Inject cadence into default combined motion**

Change the constructor to accept `aim_engine_factory: Callable[[], object] | None`
and `ai_poll_hz: float = 240.0`. When no aim factory is injected, create
`AimMovementEngine(nominal_hz=ai_poll_hz)`. Store a validated positive finite
poll rate, return `1 / ai_poll_hz` whenever AI is selected, and preserve the
existing Jitter cadence otherwise. Explicit fake factories continue to work.

- [ ] **Step 5: Run focused motion/composition tests**

Run: `python -m unittest tests.test_ai_targeting tests.test_combined_motion -v`

Expected: PASS; update fake engine construction only where the new optional
factory contract requires it.

- [ ] **Step 6: Commit the servo**

```powershell
git add ai_targeting.py combined_motion.py tests/test_ai_targeting.py tests/test_combined_motion.py
git commit -m "feat: stream AI aim through a time-based servo"
```

---

### Task 5: Conservative temporal box tracker

**Files:**
- Create: `ai_tracking.py`
- Create: `tests/test_ai_tracking.py`

**Interfaces:**
- Consumes: `Detection`, `DetectionAnalysis`, `DetectionFrameSnapshot`, `TargetSnapshot`, `AimSettings`, and `detection_aim_point` from `ai_targeting.py`.
- Produces: immutable `TrackerState`, `TrackingObservation`, and `observe_detections(state, detections, settings, *, sequence, captured_at) -> TrackingObservation`.

- [ ] **Step 1: Write crossing, ambiguity, and replacement tests**

Use helper boxes centered on named person paths and assert identity, not only
coordinates:

```python
def head_box(center_x, center_y=100, size=10):
    half = size / 2
    return Detection(
        center_x - half, center_y - half,
        center_x + half, center_y + half,
        0.9, 7,
    )

def test_crossing_does_not_follow_the_nearest_competitor(self):
    acquired = observe_detections(
        TrackerState(), (head_box(130),), AimSettings(),
        sequence=1, captured_at=1 / 60,
    )
    self.assertAlmostEqual(acquired.analysis.target.aim_x, 130)
    state = acquired.state
    observations = []
    for sequence, (person_a, person_b) in enumerate(
        ((140, 220), (150, 190), (160, 160), (175, 145), (190, 130)), 2
    ):
        result = observe_detections(
            state,
            (head_box(person_b), head_box(person_a)),
            AimSettings(), sequence=sequence, captured_at=sequence / 60,
        )
        state = result.state
        observations.append(result)
    self.assertIsNone(observations[2].analysis.target)
    self.assertEqual(observations[2].analysis.frame.selected_index, None)
    self.assertAlmostEqual(observations[-1].analysis.target.aim_x, 190)
    self.assertEqual(observations[-1].analysis.frame.selected_index, 1)

def test_replacement_waits_for_three_stable_observations(self):
    state = TrackerState()
    first = observe_detections(
        state, (head_box(100),), AimSettings(),
        sequence=1, captured_at=0.0,
    )
    state = first.state
    results = []
    for sequence, captured_at in ((2, 0.150), (3, 0.167), (4, 0.184)):
        observed = observe_detections(
            state, (head_box(220),), AimSettings(),
            sequence=sequence, captured_at=captured_at,
        )
        state = observed.state
        results.append(observed.analysis.target)
    self.assertEqual(results[:2], [None, None])
    self.assertAlmostEqual(results[2].aim_x, 220)
```

Add tests for all accepted boxes remaining in ambiguous frames, two-frame
original recovery, changed recovery reset, inclusive score margin 0.15,
plausibility radius, area ratios 0.4/2.5, velocity cap 800 px/s, prediction cap
100 ms, exact 150 ms expiry, inclusive pending displacement 18 px, head-first
initial/replacement selection, malformed/unaccepted classes, and immutable
independent states.

- [ ] **Step 2: Run tracker tests and confirm RED**

Run: `python -m unittest tests.test_ai_tracking -v`

Expected: import failure because `ai_tracking.py` does not exist.

- [ ] **Step 3: Implement candidate geometry and scoring**

Create internal immutable `_Candidate(index, detection, target)` records.
Filter by confidence/class, compute positive box area, IoU, bounded predicted
velocity, normalized distance, and bounded log-area change. Use:

```python
score = (
    0.60 * (distance / plausibility_radius)
    + 0.25 * (1.0 - iou)
    + 0.15 * min(1.0, abs(math.log(area_ratio)))
)
```

Sort ties deterministically by score, aim X, aim Y, box coordinates, then
accepted index. Two plausible candidates with score gap `<= 0.15` are
ambiguous.

- [ ] **Step 4: Implement immutable transitions and honest frames**

`TrackerState` stores confirmed detection/target, preceding target, last-clear
time, recovery candidate/count, and pending candidate/count. Never mutate an
input state. Build `DetectionFrameSnapshot` from the current accepted tuple on
every observation. Ambiguous/missing/pending results use target and selected
index `None`; a clear confirmed result indexes the exact accepted base box.

Use same-class plausible candidates first. Do not update confirmed geometry on
ambiguity. Require two consecutive clear matches after ambiguity. Expire hold
at `captured_at - last_clear_at >= 0.150`. Require three pending observations
of one class within 18 pixels before replacement. Initial acquisition is
immediate only when center-normalized best/runner gap exceeds 0.15.

- [ ] **Step 5: Run pure tracker and existing targeting tests**

Run: `python -m unittest tests.test_ai_tracking tests.test_ai_targeting -v`

Expected: PASS and the current public one-frame `analyze_detections` helpers
remain available for compatibility/tests, although AiService will stop using
them in Task 6.

- [ ] **Step 6: Commit the tracker**

```powershell
git add ai_tracking.py tests/test_ai_tracking.py
git commit -m "feat: track overlapping AI targets conservatively"
```

---

### Task 6: Generation-safe tracker and adaptive-zoom integration

**Files:**
- Modify: `ai_service.py`
- Modify: `ai_targeting.py`
- Modify: `tests/test_ai_service.py`
- Modify: `tests/test_ai_targeting.py`

**Interfaces:**
- Consumes: `TrackerState` and `observe_detections` from Task 5; existing zoom state/composition and generation checks.
- Produces: one atomic current-frame target/detection publication per base frame, with no refinement during ambiguity.

- [ ] **Step 1: Write failing service-level crossing tests**

Extend the controlled detector/capture fixtures with full boxes for two heads.
Assert the sequence:

```python
# A is initially selected; A and B converge; ambiguous frame publishes:
self.assertIsNone(service.latest_snapshot())
self.assertIsNone(service.latest_detection_snapshot().selected_index)
self.assertEqual(len(service.latest_detection_snapshot().detections), 2)
# Two clear post-crossing A frames resume A, never B.
```

Add tests that ambiguity performs one detector call even when zoom gate is
true, a genuine replacement publishes only on frame three, Jitter+AI receives
no target while pending, an old generation cannot publish tracker recovery,
and refinement success replaces only the confirmed base box as before.

- [ ] **Step 2: Run focused service tests and confirm RED**

Run: `python -m unittest -v tests.test_ai_service.AiServiceTests.test_crossing_ambiguity_withholds_target_and_refinement tests.test_ai_service.AiServiceTests.test_service_replacement_waits_for_three_clear_frames`

Expected: current point lock chooses one nearby box immediately or imports are missing.

- [ ] **Step 3: Replace per-worker point lock with tracker state**

Inside `_worker`, initialize `tracker_state = TrackerState()`. Replace base
`analyze_detections(... previous=...)`, `observe_target_lock`, and
`target_lock_allows` with:

```python
tracked = observe_detections(
    tracker_state,
    base_detections,
    settings,
    sequence=sequence,
    captured_at=captured_at,
)
tracker_state = tracked.state
base_analysis = tracked.analysis
```

Leave generation/stop checks around capture and inference in their existing
order. When selected index or target is `None`, skip zoom factor/crop/second
inference and publish the honest base analysis. Refined detections never enter
the tracker.

Remove the superseded `TargetLockState`, `observe_target_lock`,
`target_lock_allows`, and their point-radius tests from `ai_targeting.py` and
`tests/test_ai_targeting.py`. Keep `TARGET_ASSOCIATION_RADIUS_PX` only if the
one-frame compatibility selector still uses it; no service/runtime path may
retain both lock systems.

- [ ] **Step 4: Run AI service, zoom, and composition tests**

Run: `python -m unittest tests.test_ai_service tests.test_ai_zoom tests.test_combined_motion -v`

Expected: PASS; all inference-count, fallback, FPS, and stale-generation tests remain green.

- [ ] **Step 5: Commit service integration**

```powershell
git add ai_service.py ai_targeting.py tests/test_ai_service.py tests/test_ai_targeting.py
git commit -m "feat: integrate conservative tracking into AI inference"
```

---

### Task 7: Wire adaptive cadence through capture, Makcu, and runtime UI

**Files:**
- Modify: `ai_service.py`
- Modify: `makcu_service.py`
- Modify: `ui.py`
- Modify: `tests/test_ai_capture.py`
- Modify: `tests/test_ai_service.py`
- Modify: `tests/test_makcu_service.py`
- Modify: `tests/test_ui.py`

**Interfaces:**
- Consumes: `RuntimeCadence`/`detect_runtime_cadence` from Task 1 and `CombinedMotionEngine.ai_poll_hz` from Task 4.
- Produces: `AiService(..., capture_fps: int = 120)`, `MakcuService(..., ai_poll_hz: float = 240.0)`, and `JitterApp(..., runtime_cadence: RuntimeCadence | None = None)`.

- [ ] **Step 1: Write failing factory/cadence tests**

```python
def test_app_default_factories_receive_runtime_cadence(self):
    cadence = RuntimeCadence(144, 144, 288)
    app = self.make_app(runtime_cadence=cadence)
    self.assertEqual(app.runtime_cadence, cadence)
    self.assertEqual(app.ai_cadence_var.get(), "DISPLAY 144 HZ · SERVO 288 HZ")

def test_fallback_cadence_status_is_explicit(self):
    app = self.make_app(runtime_cadence=RuntimeCadence(None, 120, 240))
    self.assertEqual(app.ai_cadence_var.get(), "DISPLAY AUTO · SERVO 240 HZ")
```

Add an AiService test whose default-capture seam records `target_fps=165`, a
Makcu default combined factory test whose stop-event timeout is `1/330`, and a
DXCam test confirming its already supported `target_fps` reaches
`camera.start` without changing output index/region/color/buffer arguments.

- [ ] **Step 2: Run focused integration tests and confirm RED**

Run: `python -m unittest tests.test_ai_capture tests.test_ai_service tests.test_combined_motion tests.test_makcu_service.MakcuMovementTests.test_ai_worker_uses_injected_servo_hz tests.test_ui.JitterLayoutTests.test_app_default_factories_receive_runtime_cadence -v`

Expected: constructor/status failures because runtime cadence is not wired.

- [ ] **Step 3: Add default factory parameters while preserving injections**

In `AiService`, change the capture factory default to `None`; when absent use
`lambda: DxcamCapture(target_fps=capture_fps)`. Explicit test factories remain
authoritative. Validate capture FPS as a positive integer and fallback to 120.

In `MakcuService`, store a positive finite `ai_poll_hz`; its default combined
factory constructs `CombinedMotionEngine(..., ai_poll_hz=ai_poll_hz)`. Change
the default `aim_engine_factory` to `None` so the default combined engine can
construct `AimMovementEngine(nominal_hz=ai_poll_hz)`; an explicitly injected
aim factory remains authoritative. Explicit `combined_engine_factory` also
remains authoritative.

- [ ] **Step 4: Inject one cadence into JitterApp and render runtime status**

Add `runtime_cadence` to `JitterApp.__init__`, set
`self.runtime_cadence = runtime_cadence or detect_runtime_cadence()` before
service factories are built, and create:

```python
self.ai_cadence_var = tk.StringVar(
    self,
    (f"DISPLAY {cadence.display_hz} HZ · SERVO {cadence.servo_hz} HZ"
     if cadence.display_hz is not None
     else f"DISPLAY AUTO · SERVO {cadence.servo_hz} HZ"),
)
```

Default service factories pass capture/servo values; caller-supplied factories
keep their one-argument sink signature. Add the cadence label to AI Runtime
without changing window geometry or persisted configuration.

- [ ] **Step 5: Run capture, service, Makcu, entrypoint, and UI suites**

Run: `python -m unittest tests.test_ai_capture tests.test_ai_service tests.test_combined_motion tests.test_makcu_service tests.test_entrypoints tests.test_ui -v`

Expected: PASS, including lazy self-check imports and immediate motion cancellation.

- [ ] **Step 6: Commit adaptive runtime wiring**

```powershell
git add ai_service.py makcu_service.py ui.py tests/test_ai_capture.py tests/test_ai_service.py tests/test_makcu_service.py tests/test_ui.py
git commit -m "feat: run AI capture and motion at adaptive cadence"
```

---

### Task 8: Live five-point Tk curve editor

**Files:**
- Modify: `ui.py`
- Modify: `tests/test_ui.py`

**Interfaces:**
- Consumes: `DEFAULT_RESPONSE_CURVE`, `validated_response_curve`, `response_curve_value`, and schema-backed `AimSettings.response_curve`.
- Produces: themed `ai_curve_canvas`, four exact `ai_curve_vars`, live immutable settings updates, and `Reset Curve`.

- [ ] **Step 1: Write failing layout and interaction tests**

```python
def test_response_curve_card_is_scrollable_and_keeps_window_fixed(self):
    self.assertIs(self.app.ai_curve_card.master, self.app.motion_scroll_content)
    self.assertEqual(self.app.ai_curve_card.winfo_manager(), "grid")
    self.assertIsInstance(self.app.ai_curve_canvas, tk.Canvas)
    self.assertEqual(self.app.geometry().split("+")[0], "840x620")
    self.assertEqual(self.app.stop_button.winfo_manager(), "grid")

def test_curve_exact_edit_updates_live_snapshot_and_schedules_save(self):
    self.app._cancel_after("_save_after_id")
    self.app.ai_curve_vars[2].set("42")
    self.app._curve_entry_changed(2)
    self.assertEqual(self.app.get_ai_settings().response_curve[2], 0.42)
    self.assertIsNotNone(self.app._save_after_id)

def test_reset_curve_restores_default(self):
    self.app.ai_curve_vars[1].set("20")
    self.app._curve_entry_changed(1)
    self.app.ai_curve_reset_button.invoke()
    self.assertEqual(self.app.get_ai_settings().response_curve, DEFAULT_RESPONSE_CURVE)
```

Add tests for the fixed zero point, drag clamping between neighbors, exact
0/100 boundaries, invalid text/range/order styling without snapshot changes,
theme redraw colors, Canvas destroy safety, config restoration, scalar
`ai_vars` excluding `response_curve`, and save output containing curve but no
Canvas/runtime state. Add a Test 3s assertion that its existing
`aim_provider` returns the live curve snapshot and that adaptive zoom remains
disabled during the test.

- [ ] **Step 2: Run focused UI tests and confirm RED**

Run: `python -m unittest -v tests.test_ui.JitterLayoutTests.test_response_curve_card_is_scrollable_and_keeps_window_fixed tests.test_ui.JitterLayoutTests.test_curve_exact_edit_updates_live_snapshot_and_schedules_save tests.test_ui.JitterLayoutTests.test_reset_curve_restores_default`

Expected: missing curve card/variables/methods.

- [ ] **Step 3: Separate scalar variables from curve variables**

Build `ai_vars` only from `_AI_CONTROL_SPECS`. Create four StringVars for
indices 1..4 using whole percentages. Add `_current_ai_mapping()` that returns
the four scalar strings plus a five-element normalized curve list. Use this
helper in scalar edits, curve edits, reset, and immutable snapshot replacement.

- [ ] **Step 4: Build the full-width graph card and exact entries**

Place `ai_curve_card` below the two existing AI cards with `columnspan=2`.
Create a no-highlight themed Canvas, draw grid/axis labels, sample
`response_curve_value` across its width, and draw five nodes. Bind press/drag/
release only to adjustable node hit regions. Convert Y pixels to 0..1, clamp
between neighboring values, update the corresponding percentage StringVar,
replace the snapshot, redraw, and schedule save.

Create exact entries for 25/50/75/100% with the existing entry styles. Invalid
values set only the affected entries to `Liquid.Invalid.TEntry`, set an
actionable footer message, and leave the last valid graph/snapshot untouched.
`Reset Curve` restores `(0, .12, .35, .68, 1)`, redraws, and schedules save.

- [ ] **Step 5: Redraw on theme and teardown safely**

Call `_redraw_ai_curve()` after palette application and Canvas configure.
Catch only normal Tk teardown errors after `_closing`/widget destruction; do
not mask drawing bugs. Ensure no Canvas callback survives `close_app()`.

- [ ] **Step 6: Run all UI and settings tests**

Run: `python -m unittest tests.test_ui tests.test_settings -v`

Expected: PASS in both themes with fixed 840x620 geometry and visible STOP.

- [ ] **Step 7: Commit the curve editor**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: add a live AI response curve editor"
```

---

### Task 9: Documentation, canonical verification, and connected acceptance

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Test: all repository tests and hardware acceptance

**Interfaces:**
- Consumes: completed Tasks 1-8 and the approved design spec.
- Produces: accurate operator/developer documentation and verification evidence; no package build.

- [ ] **Step 1: Update behavior and schema documentation**

Document in README:

- conservative full-box tracking, ambiguity pause, two-frame original recovery,
  and three-frame replacement;
- five-point distance-to-speed curve, four adjustable exact percentages,
  Strength/Smoothing/Max Step interaction, and Reset Curve;
- primary-display capture cap, double-rate servo range, fallback cadence, and
  runtime status; and
- time-based microsteps, 150 ms stale discard, and unchanged Jitter/STOP behavior.

Update AGENTS planned layout with `ai_tracking.py` and `display_timing.py`, set
schema text to 5 with schema 6 reserved as unsupported future data, state that
the curve alone is persisted, describe adaptive cadence as runtime-only, and
add both modules to the canonical compile command.

- [ ] **Step 2: Run focused regression groups**

```powershell
python -m unittest tests.test_display_timing tests.test_ai_tracking tests.test_ai_targeting tests.test_ai_service tests.test_ai_zoom tests.test_ai_capture tests.test_combined_motion tests.test_makcu_service tests.test_settings tests.test_ui -v
```

Expected: PASS with no target switch, cadence, servo, schema, or UI regression.

- [ ] **Step 3: Run the canonical compile and full unit suite**

```powershell
python -m py_compile main.py ui.py motion.py combined_motion.py ai_targeting.py ai_tracking.py ai_detection.py ai_capture.py ai_zoom.py ai_service.py display_timing.py overlay.py makcu_service.py hotkeys.py settings.py sound_service.py liquid_widgets.py distribution_metadata.py
python -m unittest discover -s tests -v
```

Expected: every module compiles and the complete suite reports `OK`.

- [ ] **Step 4: Run runtime and distribution verification**

```powershell
python -c "import makcu, serial, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
git diff --check
git status --short --branch
```

Expected: imports succeed; self-check returns status `ok` with
`DmlExecutionProvider`; distribution review succeeds; diff check is empty;
only intended tracked edits are present. Do not run Nuitka.

- [ ] **Step 5: Commit documentation and verification-ready state**

```powershell
git add README.md AGENTS.md
git commit -m "docs: explain adaptive AI tracking and response curves"
```

- [ ] **Step 6: Run connected Makcu acceptance with the user**

Start the source app from this worktree. Confirm the displayed cadence matches
the primary monitor policy. In AI-only and Jitter+AI modes, fire while two
targets approach, overlap, cross, and separate; verify no competing pull,
ambiguity pauses AI only, two clear observations resume the original, and a
lost original requires three stable replacement observations. Edit near/mid/
far curve points and verify continuous movement without bursts, drift,
oscillation, or overshoot. Exercise Trigger/Modifier release, hotkey disable,
STOP, disconnect/reconnect, Test 3s, and shutdown. Close normally and inspect
new `app.log` lines for tracker, cadence, AI, Overlay, generation, or Makcu
errors. Preserve ignored `config.json`, backup, and log files unless the user
explicitly chooses cleanup.

- [ ] **Step 7: Re-run the full suite after hardware acceptance**

Run: `python -m unittest discover -s tests -v`

Expected: complete suite still reports `OK`; branch status is clean and ready
for the finishing-development-branch workflow.
