# Combined AI Aim, Jitter, and Detection Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace exclusive Jitter/AI modes with independently selectable sources that can move together through one Makcu worker, and add an independent click-through red detection overlay.

**Architecture:** Add immutable detection-frame publication to the existing AI pipeline, compose the existing Jitter and AI engines in a new pure module, and route all motion combinations through one generation-safe Makcu worker. A new main-thread-only Tk/Win32 overlay consumes the latest immutable detection snapshot, while `ui.py` owns non-persisted source, Master, and Overlay states and reconciles shared AI runtime demand.

**Tech Stack:** Python 3.12, Tkinter/ttk, ctypes Win32 APIs, NumPy, DXCam, ONNX Runtime DirectML, Makcu, and `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-27-combined-ai-jitter-overlay-design.md`

## Global Constraints

- Windows-only; keep every Tk widget, Tk variable, `Toplevel`, and canvas call on the main thread.
- Use only `models/all_games_320.onnx` with the fixed centered 320-by-320 DXCam capture.
- Add no dependency, alternate model, training, profile, tray, Pillow, Pystray, Torch, Ultralytics, or OpenCV feature.
- Keep exactly one Makcu controller owner and at most one production or Test Run motion worker.
- Preserve immediate cancellation, exact generation filtering, the final STOP barrier, fractional accumulation, acceleration limits, and discard-not-queue behavior.
- Start Jitter selection, AI selection, Master, and Overlay `False` on every launch; never persist runtime selections, targets, detections, FPS, provider, or Moving state.
- Overlay must be 320 by 320 on the primary display, topmost, no-activate, click-through, and successfully excluded from capture before becoming visible.
- Keep blocking Makcu, capture, inference, and cleanup work off the Tk event loop.
- Use TDD for every behavior: write the focused test, observe the expected failure, add the minimal implementation, then run the focused module and affected integration tests.
- Do not run Nuitka. Do not edit generated output or user data.

---

## File Map

- Create `combined_motion.py`: immutable source selection and pure composition of `PairedPulseEngine` plus `AimMovementEngine`.
- Create `overlay.py`: pure box projection plus the main-thread Tk/Win32 detection overlay.
- Create `tests/test_combined_motion.py`: composition, scheduling, clamp, zero, and factory-isolation coverage.
- Create `tests/test_overlay.py`: projection, freshness, Tk drawing, Win32 setup failure, and cleanup coverage.
- Modify `ai_targeting.py`: immutable detection-frame and analysis result types; preserve `select_target` compatibility.
- Modify `ai_service.py`: atomically publish and clear target plus detection snapshots.
- Modify `makcu_service.py`: route Jitter-only, AI-only, and combined sources through one composite worker.
- Modify `ui.py`: independent source controls, Master, demand reconciliation, overlay polling, Test Run matrix, and fallback behavior.
- Modify `settings.py`: schema 4, remove exclusive mode from `AppConfig` and serialized documents.
- Modify focused tests in `tests/test_ai_targeting.py`, `tests/test_ai_service.py`, `tests/test_makcu_service.py`, `tests/test_ui.py`, and `tests/test_settings.py`.
- Modify `distribution_metadata.py`, `tests/test_distribution_metadata.py`, and `tests/test_entrypoints.py`: recognize approved overlay source and new compile targets.
- Modify `README.md` and `AGENTS.md`: document the approved combined-source and overlay decision plus verification targets.

---

### Task 1: Publish a Pure Detection Analysis Result

**Files:**
- Modify: `ai_targeting.py:17-144`
- Modify: `tests/test_ai_targeting.py:65-128`

**Interfaces:**
- Consumes: existing `Detection`, `TargetSnapshot`, `AimSettings`, and target-selection rules.
- Produces: `DetectionFrameSnapshot`, `DetectionAnalysis`, and `analyze_detections(...)`; existing `select_target(...)` remains a target-only wrapper.

- [ ] **Step 1: Add failing immutable-frame and selection-index tests**

Add these imports and tests to `tests/test_ai_targeting.py`:

```python
from dataclasses import FrozenInstanceError

from ai_targeting import (
    AimSettings,
    Detection,
    DetectionFrameSnapshot,
    analyze_detections,
)


def test_analysis_filters_confidence_and_preserves_selected_box_index(self):
    low = Detection(1, 2, 10, 20, 0.20, 7)
    player = Detection(20, 30, 60, 130, 0.80, 0)
    head = Detection(140, 140, 180, 180, 0.90, 7)

    result = analyze_detections(
        (low, player, head),
        AimSettings(confidence=0.35),
        sequence=4,
        captured_at=10.0,
    )

    self.assertEqual(result.frame, DetectionFrameSnapshot(
        sequence=4,
        captured_at=10.0,
        detections=(player, head),
        selected_index=1,
    ))
    self.assertIsNotNone(result.target)
    self.assertEqual(result.target.target_class, "head")


def test_detection_frame_is_deeply_immutable_and_empty_analysis_is_publishable(self):
    result = analyze_detections(
        (), AimSettings(), sequence=9, captured_at=20.0
    )
    self.assertIsNone(result.target)
    self.assertEqual(result.frame.detections, ())
    self.assertIsNone(result.frame.selected_index)
    with self.assertRaises(FrozenInstanceError):
        result.frame.sequence = 10
```

Place these methods on `TargetSelectionTests` and use `self.assert*` consistently with the existing file.

- [ ] **Step 2: Run the tests and confirm the missing-interface failure**

Run:

```powershell
python -m unittest -v tests.test_ai_targeting.TargetSelectionTests
```

Expected: import failure for `DetectionFrameSnapshot` or `analyze_detections`, proving the new public result does not exist yet.

- [ ] **Step 3: Implement immutable analysis without duplicating target policy**

Add these public records and refactor candidate selection so accepted detections retain their tuple indices:

```python
@dataclass(frozen=True)
class DetectionFrameSnapshot:
    sequence: int
    captured_at: float
    detections: tuple[Detection, ...]
    selected_index: int | None


@dataclass(frozen=True)
class DetectionAnalysis:
    target: TargetSnapshot | None
    frame: DetectionFrameSnapshot


def analyze_detections(
    detections: Iterable[Detection],
    settings: AimSettings,
    *,
    sequence: int,
    captured_at: float,
    previous: TargetSnapshot | None = None,
) -> DetectionAnalysis:
    accepted = tuple(
        detection
        for detection in detections
        if detection.confidence >= settings.confidence
        and _aim_point(detection) is not None
    )
    candidates = [
        (index, point)
        for index, detection in enumerate(accepted)
        if (point := _aim_point(detection)) is not None
    ]
    heads = [item for item in candidates if item[1][0] == "head"]
    candidates = heads or [item for item in candidates if item[1][0] == "player"]
    selected_index = None
    target = None
    if candidates:
        target_class = candidates[0][1][0]
        origin = (160.0, 160.0)
        if previous is not None and previous.target_class == target_class:
            associated = [
                item for item in candidates
                if math.hypot(
                    item[1][1] - previous.aim_x,
                    item[1][2] - previous.aim_y,
                ) <= 48.0
            ]
            if associated:
                candidates = associated
                origin = (previous.aim_x, previous.aim_y)
        selected_index, selected = min(
            candidates,
            key=lambda item: math.hypot(
                item[1][1] - origin[0], item[1][2] - origin[1]
            ),
        )
        target = TargetSnapshot(sequence, captured_at, *selected)
    return DetectionAnalysis(
        target=target,
        frame=DetectionFrameSnapshot(
            sequence, captured_at, accepted, selected_index
        ),
    )


def select_target(
    detections: Iterable[Detection],
    settings: AimSettings,
    *,
    sequence: int,
    captured_at: float,
    previous: TargetSnapshot | None = None,
) -> TargetSnapshot | None:
    return analyze_detections(
        detections,
        settings,
        sequence=sequence,
        captured_at=captured_at,
        previous=previous,
    ).target
```

Use the existing full `select_target` signature in the wrapper; do not introduce a second target policy.

- [ ] **Step 4: Run targeting tests**

Run:

```powershell
python -m unittest -v tests.test_ai_targeting
```

Expected: every existing target and movement test plus the new frame tests passes.

- [ ] **Step 5: Commit the pure analysis change**

```powershell
git add -- ai_targeting.py tests/test_ai_targeting.py
git commit -m "feat: publish immutable detection analysis"
```

---

### Task 2: Publish Detection Frames from AiService

**Files:**
- Modify: `ai_service.py:12-170,228-350`
- Modify: `tests/test_ai_service.py:347-634,833-end`

**Interfaces:**
- Consumes: `analyze_detections(...)`, `DetectionFrameSnapshot`, and the existing generation-safe AI worker.
- Produces: `AiService.latest_detection_snapshot() -> DetectionFrameSnapshot | None`; target and frame publication is atomic under `_lock`.

- [ ] **Step 1: Add failing dual-publication and invalidation tests**

Add focused assertions to `AiServiceTests`:

```python
def test_worker_atomically_publishes_target_and_detection_frame(self):
    head = Detection(150, 150, 170, 170, 0.9, 7)
    service = AiService(
        lambda _event: None,
        detector_factory=lambda _path: SequenceDetector([(head,)]),
        capture_factory=lambda: FakeCapture([object()]),
    )
    self.addCleanup(service.close)

    service.start(AimSettings)

    self.assertTrue(wait_until(
        lambda: service.latest_detection_snapshot() is not None
    ))
    target = service.latest_snapshot()
    frame = service.latest_detection_snapshot()
    self.assertEqual(target.sequence, frame.sequence)
    self.assertEqual(frame.detections, (head,))
    self.assertEqual(frame.selected_index, 0)


def test_stop_error_and_old_generation_clear_or_cannot_replace_both_snapshots(self):
    # Reuse the existing blocking-detector generation fixture.
    service.stop("manual")
    self.assertIsNone(service.latest_snapshot())
    self.assertIsNone(service.latest_detection_snapshot())
```

Extend the existing inference-error and old-generation tests to assert both providers are `None` after invalidation and that an obsolete result changes neither sequence.

- [ ] **Step 2: Run the focused service tests and confirm failure**

Run:

```powershell
python -m unittest -v tests.test_ai_service.AiServiceTests.test_worker_atomically_publishes_target_and_detection_frame
```

Expected: `AttributeError` for `latest_detection_snapshot`.

- [ ] **Step 3: Implement dual snapshot storage and clearing**

Change the imports and state:

```python
from ai_targeting import (
    AimSettings,
    DetectionFrameSnapshot,
    TargetSnapshot,
    analyze_detections,
)

self._latest: TargetSnapshot | None = None
self._latest_detection: DetectionFrameSnapshot | None = None

def latest_detection_snapshot(self) -> DetectionFrameSnapshot | None:
    with self._lock:
        return self._latest_detection

def _clear_snapshots_locked(self) -> None:
    self._latest = None
    self._latest_detection = None
```

Use `_clear_snapshots_locked()` in start reset, stop, close, thread-start rollback, and `_fail_current`. In `_worker`, compute settings once for each frame and publish both values in the same locked current-generation check:

```python
analysis = analyze_detections(
    detector.detect(frame),
    settings_provider(),
    sequence=sequence,
    captured_at=captured_at,
    previous=previous,
)
with self._lock:
    if not self._is_current_locked(generation, stop_event):
        return
    self._latest = analysis.target
    self._latest_detection = analysis.frame
previous = analysis.target
```

- [ ] **Step 4: Run the complete AI service module**

Run:

```powershell
python -m unittest -v tests.test_ai_service
```

Expected: all lifecycle, barrier, FPS, target, and new detection-publication tests pass.

- [ ] **Step 5: Commit AI publication**

```powershell
git add -- ai_service.py tests/test_ai_service.py
git commit -m "feat: publish AI detection frames"
```

---

### Task 3: Build the Pure Composite Motion Engine

**Files:**
- Create: `combined_motion.py`
- Create: `tests/test_combined_motion.py`

**Interfaces:**
- Consumes: `MotionSettings`, `PairedPulseEngine`, `AimSettings`, `AimMovementEngine`, and `TargetSnapshot`.
- Produces: `MotionSources(jitter: bool, ai: bool)` and `CombinedMotionEngine.step(...) -> tuple[int, int]` plus `poll_interval(...) -> float`.

- [ ] **Step 1: Write failing composition tests**

Create `tests/test_combined_motion.py` with deterministic recording engines:

```python
import unittest

from ai_targeting import AimSettings, TargetSnapshot
from combined_motion import CombinedMotionEngine, MotionSources
from motion import MotionSettings


class FixedJitter:
    def __init__(self, report):
        self.report = report
        self.calls = []

    def step(self, settings, dt, elapsed):
        self.calls.append((settings, dt, elapsed))
        return self.report


class FixedAim:
    def __init__(self, report):
        self.report = report
        self.calls = []

    def step(self, snapshot, settings, now):
        self.calls.append((snapshot, settings, now))
        return self.report


class CombinedMotionTests(unittest.TestCase):
    def test_combines_due_components_and_clamps_final_report(self):
        jitter = FixedJitter((100, -100))
        aim = FixedAim((60, -60))
        engine = CombinedMotionEngine(
            MotionSources(jitter=True, ai=True),
            jitter_engine_factory=lambda: jitter,
            aim_engine_factory=lambda: aim,
        )
        report = engine.step(
            MotionSettings(),
            TargetSnapshot(1, 1.0, "head", 200, 200),
            AimSettings(),
            dt=0.01,
            elapsed=0.02,
            now=1.01,
        )
        self.assertEqual(report, (127, -127))
        self.assertEqual(len(jitter.calls), 1)
        self.assertEqual(len(aim.calls), 1)

    def test_jitter_continues_when_ai_has_no_target(self):
        jitter = FixedJitter((2, -1))
        aim = FixedAim((0, 0))
        engine = CombinedMotionEngine(
            MotionSources(True, True),
            jitter_engine_factory=lambda: jitter,
            aim_engine_factory=lambda: aim,
        )
        self.assertEqual(
            engine.step(
                MotionSettings(), None, AimSettings(),
                dt=0.01, elapsed=0.02, now=1.0,
            ),
            (2, -1),
        )

    def test_disabled_component_factory_is_never_constructed(self):
        engine = CombinedMotionEngine(
            MotionSources(jitter=True, ai=False),
            jitter_engine_factory=lambda: FixedJitter((1, 1)),
            aim_engine_factory=lambda: self.fail("AI factory must stay unused"),
        )
        self.assertEqual(engine.poll_interval(MotionSettings(pulse_rate_hz=50)), 0.01)
```

Add cases for AI-only output, zero output, both sources false rejection, and final excess not appearing on the next zero component tick.

- [ ] **Step 2: Run the new module and confirm import failure**

Run:

```powershell
python -m unittest -v tests.test_combined_motion
```

Expected: `ModuleNotFoundError: No module named 'combined_motion'`.

- [ ] **Step 3: Implement source validation, composition, and polling interval**

Create `combined_motion.py`:

```python
"""Pure composition of Jitter and AI Aim movement."""

from collections.abc import Callable
from dataclasses import dataclass

from ai_targeting import AimMovementEngine, AimSettings, TargetSnapshot
from motion import MotionSettings, PairedPulseEngine


@dataclass(frozen=True)
class MotionSources:
    jitter: bool = False
    ai: bool = False

    @property
    def any(self) -> bool:
        return self.jitter or self.ai


class CombinedMotionEngine:
    def __init__(
        self,
        sources: MotionSources,
        jitter_engine_factory: Callable[[], object] = PairedPulseEngine,
        aim_engine_factory: Callable[[], object] = AimMovementEngine,
    ) -> None:
        if not sources.any:
            raise ValueError("At least one motion source must be selected")
        self.sources = sources
        self._jitter = jitter_engine_factory() if sources.jitter else None
        self._aim = aim_engine_factory() if sources.ai else None

    def step(
        self,
        motion_settings: MotionSettings,
        target: TargetSnapshot | None,
        aim_settings: AimSettings,
        *,
        dt: float,
        elapsed: float,
        now: float,
    ) -> tuple[int, int]:
        jitter = (
            self._jitter.step(motion_settings, dt, elapsed)
            if self._jitter is not None else (0, 0)
        )
        aim = (
            self._aim.step(target, aim_settings, now)
            if self._aim is not None else (0, 0)
        )
        return (
            max(-127, min(127, int(jitter[0]) + int(aim[0]))),
            max(-127, min(127, int(jitter[1]) + int(aim[1]))),
        )

    def poll_interval(self, motion_settings: MotionSettings) -> float:
        if self.sources.ai:
            return 1.0 / 240.0
        rate = max(20.0, min(120.0, float(motion_settings.pulse_rate_hz)))
        return 1.0 / (rate * 2.0)
```

The composite object stores no final-clamp remainder, so clamp excess is discarded by construction.

- [ ] **Step 4: Run pure motion suites**

Run:

```powershell
python -m unittest -v tests.test_combined_motion tests.test_motion tests.test_ai_targeting
```

Expected: all pure engine tests pass.

- [ ] **Step 5: Commit the composite engine**

```powershell
git add -- combined_motion.py tests/test_combined_motion.py
git commit -m "feat: compose Jitter and AI motion"
```

---

### Task 4: Route All Makcu Motion Through the Composite Worker

**Files:**
- Modify: `makcu_service.py:17-64,420-787`
- Modify: `tests/test_makcu_service.py:49-116,532-1643`

**Interfaces:**
- Consumes: `MotionSources` and `CombinedMotionEngine` from Task 3; existing providers and motion-generation lifecycle.
- Produces: `start_composite_motion_source(sources, motion_settings_provider, target_provider, aim_settings_provider, duration_s=None) -> int | None`; old Jitter and AI start methods remain compatibility wrappers over this path.

- [ ] **Step 1: Add failing combined-worker tests**

Extend `MakcuMovementTests` with a fake composite engine and these assertions:

```python
class RecordingCombinedEngine:
    def __init__(self, sources, report=(3, -2)):
        self.sources = sources
        self.report = report
        self.calls = []

    def step(self, motion, target, aim, *, dt, elapsed, now):
        self.calls.append((motion, target, aim, dt, elapsed, now))
        return self.report

    def poll_interval(self, _motion):
        return 0.001


def test_combined_source_uses_one_worker_and_one_controller_report(self):
    engines = []
    service, controller, _events = self.connected_service(
        combined_engine_factory=lambda sources: engines.append(
            RecordingCombinedEngine(sources)
        ) or engines[-1]
    )
    source = service.start_composite_motion_source(
        MotionSources(True, True),
        MotionSettings,
        lambda: TargetSnapshot(1, time.perf_counter(), "head", 170, 160),
        AimSettings,
        duration_s=0.02,
    )
    self.assertIsInstance(source, int)
    service.join_motion(1.0)
    self.assertEqual(engines[0].sources, MotionSources(True, True))
    self.assertTrue(controller.moves)
    self.assertTrue(all(move == (3, -2) for move in controller.moves))


def test_composite_rejects_empty_sources_without_reserving_generation(self):
    service, _controller, _events = self.connected_service()
    before = service.motion_generation
    self.assertIsNone(service.start_composite_motion_source(
        MotionSources(), MotionSettings, lambda: None, AimSettings
    ))
    self.assertEqual(service.motion_generation, before)
```

Also assert legacy `start_motion_source` and `start_ai_motion_source` construct `MotionSources(True, False)` and `MotionSources(False, True)` respectively.

- [ ] **Step 2: Run focused tests and confirm missing API failure**

Run:

```powershell
python -m unittest -v tests.test_makcu_service.MakcuMovementTests.test_combined_source_uses_one_worker_and_one_controller_report
```

Expected: `AttributeError` for `start_composite_motion_source` or constructor rejection of `combined_engine_factory`.

- [ ] **Step 3: Add the composite start API and compatibility wrappers**

Update constructor injection and public entry points:

```python
from combined_motion import CombinedMotionEngine, MotionSources
from motion import MotionSettings, PairedPulseEngine

def __init__(
    self,
    event_sink: Callable[[ServiceEvent], None],
    controller_factory: Callable[..., Any] = create_controller,
    engine_factory: Callable[[], Any] = PairedPulseEngine,
    aim_engine_factory: Callable[[], Any] = AimMovementEngine,
    combined_engine_factory: Callable[[MotionSources], Any] | None = None,
) -> None:
    self._combined_engine_factory = combined_engine_factory or (
        lambda sources: CombinedMotionEngine(
            sources,
            jitter_engine_factory=engine_factory,
            aim_engine_factory=aim_engine_factory,
        )
    )

def start_composite_motion_source(
    self,
    sources: MotionSources,
    motion_settings_provider: Callable[[], MotionSettings],
    target_provider: Callable[[], TargetSnapshot | None],
    aim_settings_provider: Callable[[], AimSettings],
    duration_s: float | None = None,
) -> int | None:
    if not isinstance(sources, MotionSources) or not sources.any:
        return None
    return self._start_motion_job(
        sources,
        motion_settings_provider,
        target_provider,
        aim_settings_provider,
        duration_s,
    )
```

Make the old starts call this method with fixed sources and safe default providers. Do not leave a second mode-specific worker path.

- [ ] **Step 4: Refactor `_start_motion_job` and `_motion_worker`**

Replace the `mode` branch with one call:

```python
engine = self._combined_engine_factory(sources)

motion_settings = motion_settings_provider()
tick_started = time.perf_counter()
dt = max(0.0, min(tick_started - previous_tick, 0.1))
previous_tick = tick_started
report_x, report_y = engine.step(
    motion_settings,
    target_provider(),
    aim_settings_provider(),
    dt=dt,
    elapsed=elapsed,
    now=tick_started,
)
interval = engine.poll_interval(motion_settings)
stop_event.wait(max(
    0.0, interval - (time.perf_counter() - tick_started)
))
```

Keep the existing lock checks, `_move_barrier`, duration boundary, terminal-event reservation, and exact generation predicates byte-for-byte where their semantics do not change.

- [ ] **Step 5: Run all Makcu service tests**

Run:

```powershell
python -m unittest -v tests.test_makcu_service
```

Expected: combined tests and every existing cancellation, reconnect, callback, disconnect, and lifecycle race test pass.

- [ ] **Step 6: Commit the single-worker integration**

```powershell
git add -- makcu_service.py tests/test_makcu_service.py
git commit -m "feat: run combined motion through Makcu"
```

---

### Task 5: Add the Click-Through Capture-Excluded Overlay

**Files:**
- Create: `overlay.py`
- Create: `tests/test_overlay.py`

**Interfaces:**
- Consumes: `DetectionFrameSnapshot` from Task 1 and a Tk parent.
- Produces: `OverlayBox`, `project_overlay_boxes(snapshot, now)`, `Win32OverlayAdapter.configure(hwnd)`, and `DetectionOverlay.show/render/clear/hide/close`.

- [ ] **Step 1: Write failing pure projection tests**

Create `tests/test_overlay.py`:

```python
import unittest

from ai_targeting import Detection, DetectionFrameSnapshot
from overlay import OverlayBox, project_overlay_boxes


class OverlayProjectionTests(unittest.TestCase):
    def test_projects_all_boxes_and_emphasizes_selected_index(self):
        frame = DetectionFrameSnapshot(
            3,
            10.0,
            (
                Detection(1, 2, 30, 40, 0.8, 0),
                Detection(100, 110, 130, 150, 0.9, 7),
            ),
            1,
        )
        self.assertEqual(
            project_overlay_boxes(frame, now=10.1),
            (
                OverlayBox(1, 2, 30, 40, 2),
                OverlayBox(100, 110, 130, 150, 4),
            ),
        )

    def test_absent_or_stale_frame_projects_no_boxes(self):
        frame = DetectionFrameSnapshot(1, 10.0, (), None)
        self.assertEqual(project_overlay_boxes(None, 10.0), ())
        self.assertEqual(project_overlay_boxes(frame, 10.151), ())
```

- [ ] **Step 2: Add failing window-adapter and lifecycle tests**

Use fake window/canvas objects and a recording adapter:

```python
class FakeWindow:
    def __init__(self, calls):
        self.calls = calls
        self.destroyed = False

    def withdraw(self): self.calls.append("withdraw")
    def overrideredirect(self, value): self.calls.append(("borderless", value))
    def attributes(self, name, value): self.calls.append((name, value))
    def winfo_screenwidth(self): return 1920
    def winfo_screenheight(self): return 1080
    def geometry(self, value): self.calls.append(("geometry", value))
    def update_idletasks(self): self.calls.append("update-idletasks")
    def winfo_id(self): return 1234
    def deiconify(self): self.calls.append("deiconify")
    def lift(self): self.calls.append("lift")
    def destroy(self):
        self.destroyed = True
        self.calls.append("destroy")


class FakeCanvas:
    def __init__(self, window, options):
        self.window = window
        self.options = options
        self.items = []

    def pack(self, **options): self.pack_options = options
    def delete(self, tag): self.items = []
    def create_rectangle(self, *coords, **options):
        self.items.append((coords, options))


class RecordingAdapter:
    def __init__(self, calls, error=None):
        self.calls = calls
        self.error = error
        self.handles = []

    def configure(self, hwnd):
        self.calls.append("configure-win32")
        self.handles.append(hwnd)
        if self.error:
            raise self.error


def test_show_requires_win32_configuration_before_deiconify(self):
    calls = []
    adapter = RecordingAdapter(calls)
    window = FakeWindow(calls)
    overlay = DetectionOverlay(
        root,
        window_factory=lambda _root: window,
        canvas_factory=lambda window, **options: FakeCanvas(window, options),
        win32_adapter=adapter,
    )
    overlay.show()
    self.assertEqual(adapter.handles, [window.winfo_id()])
    self.assertLess(calls.index("configure-win32"), calls.index("deiconify"))


def test_setup_failure_destroys_window_and_remains_hidden(self):
    adapter = RecordingAdapter(
        [], OverlaySetupError("capture exclusion failed")
    )
    with self.assertRaises(OverlaySetupError):
        overlay.show()
    self.assertTrue(window.destroyed)
    self.assertFalse(overlay.visible)
```

Add render replacement, `#ff2b2b` outline, width, hide-clear-withdraw, idempotent close, and main-thread ownership tests.

- [ ] **Step 3: Run the new tests and confirm module failure**

Run:

```powershell
python -m unittest -v tests.test_overlay
```

Expected: `ModuleNotFoundError: No module named 'overlay'`.

- [ ] **Step 4: Implement pure projection and Win32 adapter**

Create these core declarations in `overlay.py`:

```python
OVERLAY_SIZE = 320
OVERLAY_COLOR = "#ff2b2b"
MAX_FRAME_AGE_S = 0.150
WDA_EXCLUDEFROMCAPTURE = 0x00000011
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000


class OverlaySetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class OverlayBox:
    x1: float
    y1: float
    x2: float
    y2: float
    width: int


def project_overlay_boxes(snapshot, now):
    if snapshot is None or max(0.0, now - snapshot.captured_at) > MAX_FRAME_AGE_S:
        return ()
    return tuple(
        OverlayBox(
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            4 if index == snapshot.selected_index else 2,
        )
        for index, detection in enumerate(snapshot.detections)
    )
```

`Win32OverlayAdapter.configure(hwnd)` must OR all four required extended styles through `GetWindowLongPtrW/SetWindowLongPtrW`, then require a truthy `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)`. Raise `OverlaySetupError` with the failing operation and `ctypes.get_last_error()`; do not silently degrade to a capturable overlay.

- [ ] **Step 5: Implement `DetectionOverlay` with lazy Tk creation**

Use this public behavior:

```python
class DetectionOverlay:
    def __init__(
        self,
        root,
        *,
        window_factory=tk.Toplevel,
        canvas_factory=tk.Canvas,
        win32_adapter=None,
        transparent_key="#010203",
    ) -> None:
        self._root = root
        self._window_factory = window_factory
        self._canvas_factory = canvas_factory
        self._win32 = win32_adapter or Win32OverlayAdapter()
        self._transparent_key = transparent_key
        self._window = None
        self._canvas = None
        self._visible = False
        self._closed = False

    @property
    def visible(self) -> bool:
        return self._visible

    def show(self) -> None:
        if self._closed:
            raise OverlaySetupError("Overlay is closed")
        if self._window is None:
            window = self._window_factory(self._root)
            window.withdraw()
            window.overrideredirect(True)
            window.attributes("-topmost", True)
            window.attributes("-transparentcolor", self._transparent_key)
            left = (window.winfo_screenwidth() - OVERLAY_SIZE) // 2
            top = (window.winfo_screenheight() - OVERLAY_SIZE) // 2
            window.geometry(f"{OVERLAY_SIZE}x{OVERLAY_SIZE}+{left}+{top}")
            canvas = self._canvas_factory(
                window,
                width=OVERLAY_SIZE,
                height=OVERLAY_SIZE,
                background=self._transparent_key,
                highlightthickness=0,
            )
            canvas.pack(fill="both", expand=True)
            window.update_idletasks()
            try:
                self._win32.configure(int(window.winfo_id()))
            except Exception:
                window.destroy()
                raise
            self._window = window
            self._canvas = canvas
        self._window.deiconify()
        self._window.lift()
        self._visible = True

    def render(self, snapshot, *, now: float) -> None:
        boxes = project_overlay_boxes(snapshot, now)
        self.clear()
        for box in boxes:
            self._canvas.create_rectangle(
                box.x1, box.y1, box.x2, box.y2,
                outline=OVERLAY_COLOR,
                width=box.width,
                tags=("detection",),
            )

    def clear(self) -> None:
        if self._canvas is not None:
            self._canvas.delete("detection")

    def hide(self) -> None:
        self.clear()
        if self._window is not None:
            self._window.withdraw()
        self._visible = False

    def close(self) -> None:
        self.clear()
        window, self._window, self._canvas = self._window, None, None
        self._visible = False
        self._closed = True
        if window is not None:
            window.destroy()
```

Use dependency-injected window, canvas, clock-independent rendering, and Win32 adapter seams so the unit tests do not need live capture or Makcu.

- [ ] **Step 6: Run overlay and targeting suites**

Run:

```powershell
python -m unittest -v tests.test_overlay tests.test_ai_targeting
```

Expected: projection, setup failure, lifecycle, and targeting tests pass without opening a persistent overlay.

- [ ] **Step 7: Commit the overlay component**

```powershell
git add -- overlay.py tests/test_overlay.py
git commit -m "feat: add detection overlay component"
```

---

### Task 6: Replace Exclusive UI Modes with Source, Master, and Overlay State

**Files:**
- Modify: `ui.py:20-90,179-295,748-798,1072-1271,1491-1750,2139-2831,3172-3236`
- Modify: `tests/test_ui.py:284-end`

**Interfaces:**
- Consumes: `MotionSources`, `DetectionOverlay`, `AiService.latest_detection_snapshot()`, and `MakcuService.start_composite_motion_source(...)`.
- Produces: independent `jitter_selected`, `ai_selected`, `master_armed`, and `overlay_visible` states; centralized `_reconcile_ai_runtime`; source-aware production and Test Run lifecycle.

- [ ] **Step 1: Replace mode-layout expectations with failing launch-state tests**

Update the UI test factory to inject a `StubOverlay` and add:

```python
class StubOverlay:
    def __init__(self):
        self.shown = self.hidden = self.closed = 0
        self.rendered = []
        self.show_error = None

    def show(self):
        if self.show_error is not None:
            raise self.show_error
        self.shown += 1
    def render(self, snapshot, *, now): self.rendered.append((snapshot, now))
    def clear(self): pass
    def hide(self): self.hidden += 1
    def close(self): self.closed += 1


def test_launches_with_all_runtime_switches_off_and_no_mode_selector(self):
    self.assertFalse(self.app.jitter_selected)
    self.assertFalse(self.app.ai_selected)
    self.assertFalse(self.app.master_armed)
    self.assertFalse(self.app.overlay_visible)
    self.assertEqual(self.app.master_button.cget("text"), "Enable Selected")
    self.assertEqual(self.app.jitter_source_button.cget("text"), "Jitter OFF")
    self.assertEqual(self.app.ai_source_button.cget("text"), "AI Aim OFF")
    self.assertFalse(hasattr(self.app, "mode_combo"))
```

Delete assertions that require the exclusive Mode combobox or one hidden settings card; replace them with assertions that both settings sections are inside the Advanced scroll.

- [ ] **Step 2: Run the layout test and confirm missing-state failure**

Run:

```powershell
python -m unittest -v tests.test_ui.JitterLayoutTests.test_launches_with_all_runtime_switches_off_and_no_mode_selector
```

Expected: `AttributeError` for a new runtime state or source button.

- [ ] **Step 3: Add runtime state, injection seams, and controls**

Update imports and constructor:

```python
from combined_motion import MotionSources
from overlay import DetectionOverlay, OverlaySetupError

def __init__(
    self,
    *,
    config_store: ConfigStore | None = None,
    service_factory: Callable[[Callable[[Any], None]], Any] | None = None,
    ai_service_factory: Callable[[Callable[[AiEvent], None]], Any] | None = None,
    hotkey_factory: Callable[[int, Callable[[], None]], Any] | None = None,
    overlay_factory: Callable[[tk.Misc], Any] | None = None,
    sound_player: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
    auto_start: bool = True,
) -> None:
    self.jitter_selected = False
    self.ai_selected = False
    self.master_armed = False
    self.overlay_visible = False
    self._clock = clock
    self._overlay_after_id = None
    self.overlay = (overlay_factory or DetectionOverlay)(self)
```

Remove `_MODE_LABELS`, `mode_var`, `mode_display_var`, Mode combobox bindings, `_mode_selected`, `on_mode_changed`, and `_show_mode_panels`. Build `jitter_source_button`, `ai_source_button`, `master_button`, and `overlay_button`; keep `self.enable_button = self.master_button` only as a short compatibility alias if non-UI integrations still use it.

- [ ] **Step 4: Add failing Master and source-transition tests**

Add these focused runtime cases:

```python
def test_master_rejects_empty_selection_and_disconnected_device(self):
    self.app.toggle_master()
    self.assertFalse(self.app.master_armed)
    self.assertIn("Select Jitter or AI Aim", self.app.footer_var.get())
    self.app.toggle_jitter_source()
    self.app.toggle_master()
    self.assertFalse(self.app.master_armed)
    self.assertEqual(self.app.footer_var.get(), "Makcu device is not connected")


def test_both_selected_start_one_combined_worker_when_gate_activates(self):
    self.service.connected = True
    self.app.toggle_jitter_source()
    self.app.toggle_ai_source()
    self.app.toggle_master()
    self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertEqual(
        self.service.composite_motion_calls[-1].sources,
        MotionSources(True, True),
    )
    self.assertTrue(self.app.master_armed)


def test_source_change_cancels_exact_generation_before_restart(self):
    self.service.connected = True
    self.app.toggle_jitter_source()
    self.app.set_master(True)
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    retiring = self.service.active_motion_generation

    self.app.toggle_ai_source()
    self.assertEqual(self.service.cancel_reasons[-1], "sources_changed")
    self.app.handle_service_event(ServiceEvent(
        "motion_stopped", "sources_changed", retiring + 99
    ))
    self.assertEqual(len(self.service.composite_motion_calls), 1)
    self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
    self.app.handle_service_event(ServiceEvent(
        "motion_stopped", "sources_changed", retiring
    ))
    self.assertEqual(
        self.service.composite_motion_calls[-1].sources,
        MotionSources(True, True),
    )
```

Implement the last test with the existing stub service's generation recorder and `_DeferredMotionAction` pattern; assert a stale different source does not consume the deferred restart.

- [ ] **Step 5: Implement source selection and Master lifecycle**

Add these core helpers and route the existing hotkey to `toggle_master`:

```python
def _selected_sources(self) -> MotionSources:
    return MotionSources(self.jitter_selected, self.ai_selected)

def toggle_master(self) -> None:
    if self._motion_mode in _TEST_MOTION_MODES:
        self.footer_var.set("Test Run is active; use STOP to cancel")
        return
    self.set_master(not self.master_armed)

def set_master(self, armed: bool) -> None:
    if not armed:
        self.master_armed = False
        self._stop_motion_runtime("Disabled by user")
        self._reconcile_ai_runtime("Master disabled")
        self._render_runtime_controls()
        return
    sources = self._selected_sources()
    if not sources.any:
        self.footer_var.set("Select Jitter or AI Aim first")
        return
    if not self.service.connected:
        self.footer_var.set("Makcu device is not connected")
        return
    self.master_armed = True
    self.trigger_gate.clear()
    self._reconcile_ai_runtime("Master enabled")
    self._render_runtime_controls()

def _request_motion_start(self, sources, duration_s=None):
    source = self.service.start_composite_motion_source(
        sources,
        self.get_motion_settings,
        self.ai_service.latest_snapshot,
        self.get_ai_settings,
        duration_s=duration_s,
    )
    if source is None or source is False:
        return False
    self._expected_motion_generation = source
    return True
```

Source toggle methods update button text, stop/restart through the exact retiring generation when armed, and automatically call `set_master(False)` after removing the final selected source.

Update `StubService` before these tests so it records the new call shape:

```python
def start_composite_motion_source(
    self, sources, motion_provider, target_provider, aim_provider,
    duration_s=None,
):
    self.started += 1
    call = SimpleNamespace(
        sources=sources,
        motion_provider=motion_provider,
        target_provider=target_provider,
        aim_provider=aim_provider,
        duration_s=duration_s,
    )
    self.composite_motion_calls.append(call)
    if not self.connected:
        return None
    if self.motion_active:
        return self.active_motion_generation
    self.motion_generation += 1
    self.active_motion_generation = self.motion_generation
    self.motion_active = True
    return self.active_motion_generation
```

Initialize `self.composite_motion_calls = []` in `StubService.__init__`, add
`latest_detection_snapshot()` to `StubAiService`, and have it return a dedicated
`self.detection_snapshot` test value.

- [ ] **Step 6: Add failing AI-demand and Overlay tests**

```python
def test_overlay_only_starts_ai_without_makcu_and_hiding_stops_final_demand(self):
    self.service.connected = False
    self.app.toggle_overlay()
    self.assertTrue(self.app.overlay_visible)
    self.assertEqual(self.overlay.shown, 1)
    self.assertEqual(len(self.ai.start_calls), 1)
    self.app.toggle_overlay()
    self.assertFalse(self.app.overlay_visible)
    self.assertEqual(self.overlay.hidden, 1)
    self.assertEqual(self.ai.stop_calls[-1], "Overlay disabled")


def test_hiding_overlay_does_not_reload_or_stop_armed_ai(self):
    self.service.connected = True
    self.app.toggle_ai_source()
    self.app.set_master(True)
    self.app.toggle_overlay()
    starts = len(self.ai.start_calls)
    self.app.toggle_overlay()
    self.assertEqual(len(self.ai.start_calls), starts)
    self.assertEqual(self.ai.stop_calls, [])


def test_overlay_setup_failure_turns_off_only_overlay(self):
    self.overlay.show_error = OverlaySetupError("affinity failed")
    with self.assertLogs(level="ERROR"):
        self.app.toggle_overlay()
    self.assertFalse(self.app.overlay_visible)
    self.assertTrue(self.app.ai_selected)
```

- [ ] **Step 7: Implement centralized AI demand and overlay polling**

```python
def _ai_runtime_required(self) -> bool:
    return (
        self.overlay_visible
        or (self.master_armed and self.ai_selected)
        or self._motion_mode in {"test_ai_loading", "test_ai", "test_combined_loading", "test_combined"}
    )

def _reconcile_ai_runtime(self, context: str) -> bool:
    required = self._ai_runtime_required()
    if required and not self._ai_runtime_active:
        return self._start_ai_runtime(context)
    if not required and self._ai_runtime_active:
        self._stop_ai_runtime(context)
        self._ai_ready = False
        self._ai_provider = None
        self._ai_runtime_active = False
    return True

def _poll_overlay(self) -> None:
    self._overlay_after_id = None
    if self._closing or not self.overlay_visible:
        return
    try:
        self.overlay.render(
            self.ai_service.latest_detection_snapshot(), now=self._clock()
        )
    except Exception:
        logging.exception("Detection overlay rendering failed")
        self._disable_overlay_after_error()
        return
    self._overlay_after_id = self.after(16, self._poll_overlay)
```

`toggle_overlay()` must call `overlay.show()` before setting the visible state, catch `OverlaySetupError`, reconcile demand after both enable and disable, and never require Makcu.

- [ ] **Step 8: Replace mode-based Test Run with a failing source matrix**

Add table-driven tests:

```python
def test_test_run_uses_selected_source_matrix(self):
    for sources in (
        MotionSources(True, False),
        MotionSources(False, True),
        MotionSources(True, True),
    ):
        with self.subTest(sources=sources):
            app = self.make_app()
            self.service.connected = True
            app.jitter_selected = sources.jitter
            app.ai_selected = sources.ai
            app.start_test_run()
            if sources.ai:
                app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
            self.assertEqual(
                self.service.composite_motion_calls[-1].sources,
                sources,
            )
            self.assertEqual(
                self.service.composite_motion_calls[-1].duration_s,
                3.0,
            )

def test_test_run_rejects_no_sources(self):
    self.service.connected = True
    self.app.start_test_run()
    self.assertIsNone(self.app._motion_mode)
    self.assertIn("Select Jitter or AI Aim", self.app.footer_var.get())
```

Update Test Run state names to source-based states, start its three-second duration only after AI Ready, preserve prior Master, and restore only after `duration_complete`. STOP or disconnect must not restore Master.

- [ ] **Step 9: Add failure, disconnect, STOP, hotkey, and close tests**

```python
def test_ai_error_falls_back_to_jitter_when_both_were_armed(self):
    self.prepare_armed_sources(MotionSources(True, True), gate_active=True)
    self.app.handle_ai_event(AiEvent("error", "RuntimeError: AI service failed"))
    self.assertTrue(self.app.jitter_selected)
    self.assertFalse(self.app.ai_selected)
    self.assertFalse(self.app.overlay_visible)
    self.assertTrue(self.app.master_armed)
    self.assertEqual(
        self.service.composite_motion_calls[-1].sources,
        MotionSources(True, False),
    )

def test_disconnect_disarms_motion_but_keeps_overlay_demand(self):
    self.app.toggle_overlay()
    self.prepare_armed_sources(MotionSources(False, True))
    self.app.handle_service_event(ServiceEvent("disconnected", "Device lost"))
    self.assertFalse(self.app.master_armed)
    self.assertTrue(self.app.overlay_visible)
    self.assertTrue(self.app._ai_runtime_active)

def test_stop_hides_overlay_but_preserves_source_choices(self):
    self.prepare_armed_sources(MotionSources(True, True))
    self.app.toggle_overlay()
    self.app.emergency_stop("Stopped by user")
    self.assertFalse(self.app.master_armed)
    self.assertFalse(self.app.overlay_visible)
    self.assertTrue(self.app.jitter_selected)
    self.assertTrue(self.app.ai_selected)

def test_close_cancels_overlay_poll_and_closes_overlay(self):
    self.app.toggle_overlay()
    callback = self.app._overlay_after_id
    self.app.close_app()
    self.assertIn(callback, self.cancelled_callbacks)
    self.assertEqual(self.overlay.closed, 1)
```

Update hotkey tests to assert Master parity, rejected empty selection, sound only after successful Master state changes, and stale epoch rejection after STOP/reconnect.

- [ ] **Step 10: Implement final lifecycle transitions and remove mode branches**

Use `master_armed` instead of `enabled` as the authoritative state; retain a read-only `enabled` compatibility property only if tests or integrations require it. `emergency_stop` hides Overlay, cancels its `after` callback, clears the gate, stops motion/Test Run, reconciles AI demand, and preserves source choices. Disconnect uses a dedicated transition that disarms without hiding Overlay. AI error deselects AI, hides Overlay, cancels the old generation, and defers an exact Jitter-only restart when applicable.

Remove all remaining `mode_var`, `mode_display_var`, `mode_combo`, `_MODE_LABELS`, and `mode_change` branches. Replace mode matrices in `tests/test_ui.py` with `MotionSources` matrices and exact source-generation assertions.

- [ ] **Step 11: Run the complete UI suite**

Run:

```powershell
python -m unittest -v tests.test_ui
```

Expected: all layout, palette, scroll, queue, lifecycle, Test Run, hotkey, STOP, reconnect, AI error, overlay, and shutdown tests pass.

- [ ] **Step 12: Commit the UI state-machine migration**

```powershell
git add -- ui.py tests/test_ui.py
git commit -m "feat: add independent AI Jitter and overlay controls"
```

---

### Task 7: Migrate Configuration to Schema 4 Without Runtime State

**Files:**
- Modify: `settings.py:26-46,123-225`
- Modify: `tests/test_settings.py:83-410`
- Modify: `ui.py:3172-3200`
- Modify: `tests/test_ui.py:3091-3125`

**Interfaces:**
- Consumes: Task 6 UI no longer reads a persisted mode.
- Produces: `SCHEMA_VERSION = 4`; `AppConfig` contains settings only and serialized schema 4 omits `mode` and all runtime state.

- [ ] **Step 1: Write failing schema-4 migration and serialization tests**

```python
def test_schema_three_preserves_settings_but_drops_legacy_mode(self):
    document = {
        "schema_version": 3,
        "mode": "ai_aim",
        "ai": {"confidence": "0.8"},
        "motion": {
            "pulse_size_px": "4",
            "pulse_rate_hz": "50",
            "ramp_mode": "Instant",
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        outcome = ConfigStore(path).load()
    self.assertEqual(outcome.config.ai.confidence, 0.8)
    self.assertEqual(outcome.config.motion.pulse_size_px, 4.0)
    self.assertFalse(hasattr(outcome.config, "mode"))


def test_schema_four_save_omits_every_runtime_state(self):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        ConfigStore(path).save(AppConfig())
        document = json.loads(path.read_text(encoding="utf-8"))
    self.assertEqual(document["schema_version"], 4)
    for key in (
        "mode", "jitter_selected", "ai_selected", "master_armed",
        "overlay_visible", "target", "detections", "fps", "provider",
    ):
        self.assertNotIn(key, document)
```

Update UI save tests to assert only `motion`, `ai`, bindings, hotkey, preset, theme, and sound values reach `AppConfig`.

- [ ] **Step 2: Run settings tests and confirm the schema/mode failure**

Run:

```powershell
python -m unittest -v tests.test_settings.ConfigStoreTests.test_schema_three_preserves_settings_but_drops_legacy_mode tests.test_settings.ConfigStoreTests.test_schema_four_save_omits_every_runtime_state
```

Expected: `SCHEMA_VERSION` is still 3 and `AppConfig.mode` still exists.

- [ ] **Step 3: Implement schema 4 and remove mode**

Make these structural changes:

```python
SCHEMA_VERSION = 4

@dataclass(frozen=True)
class AppConfig:
    motion: MotionSettings = field(default_factory=MotionSettings)
    ai: AimSettings = field(default_factory=AimSettings)
    trigger: str = "Left"
    modifier: str = "None"
    hotkey_vk: int = 0xBD
    hotkey_name: str = "-"
    selected_preset: str = "Balanced"
    theme: str = "light"
    sound_enabled: bool = True
    sound_volume: int = 70
```

Remove `VALID_MODES`, ignore schema 3's `mode` key, load AI settings for schemas 3 and 4, and remove `mode` from the saved document. Remove the temporary `mode="jitter"` construction from `ui.save_config()` and all `AppConfig(mode=...)` test fixtures.

- [ ] **Step 4: Run settings and UI suites**

Run:

```powershell
python -m unittest -v tests.test_settings tests.test_ui
```

Expected: schema migration, atomic writes, unsupported future-schema behavior, UI saves, and launch-off runtime behavior pass.

- [ ] **Step 5: Commit schema migration**

```powershell
git add -- settings.py ui.py tests/test_settings.py tests/test_ui.py
git commit -m "feat: remove persisted exclusive AI mode"
```

---

### Task 8: Update Distribution Review and Project Documentation

**Files:**
- Modify: `distribution_metadata.py:31-37`
- Modify: `tests/test_distribution_metadata.py:245-270`
- Modify: `tests/test_entrypoints.py:319-326,516-525`
- Modify: `README.md:15-48,94-112`
- Modify: `AGENTS.md:14-33,75-87,137-151`

**Interfaces:**
- Consumes: final module names and approved scope from Tasks 1-7.
- Produces: canonical review recognizes `combined_motion.py` and `overlay.py`, while still rejecting every unapproved upstream feature.

- [ ] **Step 1: Write failing distribution expectations**

Update expected compile targets:

```python
expected_compile_targets = {
    "main.py", "ui.py", "motion.py", "combined_motion.py",
    "ai_targeting.py", "ai_detection.py", "ai_capture.py",
    "ai_service.py", "overlay.py", "makcu_service.py", "hotkeys.py",
    "settings.py", "liquid_widgets.py", "distribution_metadata.py",
}
```

Remove only `overlay` and `overlays` from the prohibited approved-source token sets in `distribution_metadata.py` and `tests/test_entrypoints.py`. Keep `training`, `profile`, `profiles`, `tray`, and `ai_tracker` prohibited.

- [ ] **Step 2: Run distribution tests and confirm expected target/token failures**

Run:

```powershell
python -m unittest -v tests.test_distribution_metadata tests.test_entrypoints
```

Expected before metadata changes: the approved `overlay.py` token is rejected or compile-target expectations differ.

- [ ] **Step 3: Update metadata policy and user documentation**

Update `_PROHIBITED_SOURCE_TOKENS` to:

```python
_PROHIBITED_SOURCE_TOKENS = {
    "training", "profile", "profiles", "tray", "ai_tracker"
}
```

Document in `README.md`:

- Jitter and AI source buttons may be selected independently.
- Master and the global hotkey arm selected sources.
- Combined movement sums current source deltas; Jitter continues without a target.
- Overlay is independent, centered 320 by 320, red, click-through, capture-excluded, and starts off.
- Test 3s follows selected sources and STOP cancels immediately.

Update `AGENTS.md` to add `combined_motion.py` and `overlay.py`, replace exclusive/no-overlay guidance with the approved constrained overlay behavior, and add both files to the verification compile command. Do not weaken the bans on other upstream features or dependencies.

- [ ] **Step 4: Run metadata review tests and command**

Run:

```powershell
python -m unittest -v tests.test_distribution_metadata tests.test_entrypoints
python .\distribution_metadata.py --review-json
```

Expected: tests pass and review JSON includes both new source files in `compile_targets` without changing pinned dependencies, model hash, release materials, or Nuitka confirmation behavior.

- [ ] **Step 5: Commit metadata and docs**

```powershell
git add -- distribution_metadata.py tests/test_distribution_metadata.py tests/test_entrypoints.py README.md AGENTS.md
git commit -m "docs: document combined motion and overlay"
```

---

### Task 9: Full Verification and Real Hardware Acceptance

**Files:**
- Verify only; modify production files only through a new failing regression test if verification reveals a defect.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: fresh automated evidence plus explicit Makcu and on-screen acceptance results.

- [ ] **Step 1: Run whitespace and syntax verification**

Run:

```powershell
git diff --check
python -m py_compile main.py ui.py motion.py combined_motion.py ai_targeting.py ai_detection.py ai_capture.py ai_service.py overlay.py makcu_service.py hotkeys.py settings.py sound_service.py liquid_widgets.py distribution_metadata.py
```

Expected: both commands exit 0 with no syntax or whitespace errors.

- [ ] **Step 2: Run the complete automated suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: every test passes with zero failures and zero errors. Record the exact test count in the handoff.

- [ ] **Step 3: Verify imports, DirectML, and canonical review**

Run:

```powershell
python -c "import makcu, serial, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
```

Expected: imports exit 0; self-check reports the approved model hash and `DmlExecutionProvider`; review JSON succeeds and names `combined_motion.py` plus `overlay.py`.

- [ ] **Step 4: Run controlled Makcu hardware checks**

With COM3 connected and the pointer positioned over a safe empty area:

1. Select Jitter only, arm Master, hold Trigger/Modifier, and confirm Jitter movement.
2. Select AI only, display an approved on-screen target, arm Master, hold the gate, and confirm AI movement.
3. Select both, hold the gate, and confirm target movement has Jitter superimposed.
4. Remove the valid target while keeping the gate held and confirm Jitter continues.
5. In every source combination, press STOP during movement and confirm immediate cancellation.
6. Toggle Master twice with Mouse5 and confirm one state change per press.
7. Run Test 3s for all three source combinations and interrupt one combined run with STOP.
8. Disconnect Makcu during combined movement and confirm Master disarms without a stale report; note that physical auto-reconnect remains a separately known defect.

- [ ] **Step 5: Run real overlay acceptance**

1. Turn Overlay on while Makcu is disconnected and AI Aim is not selected; confirm inference runs without movement.
2. Display player and head targets inside the centered capture area; confirm every accepted object has a red box and the selected target is thicker.
3. Confirm clicks pass through the overlay and it never takes focus.
4. Remove targets and confirm boxes clear within the 150-millisecond freshness limit.
5. Inspect a captured DXCam frame through a temporary read-only diagnostic or test seam and confirm no red overlay rectangles appear in the capture.
6. Turn Overlay off, press STOP, and close the app; confirm no overlay window, AI worker, Makcu worker, or `main.py` process remains.

- [ ] **Step 6: Inspect final repository state**

Run:

```powershell
git status --short
git log --oneline -12
```

Expected: no uncommitted source changes. Do not run `gen.bat`, Nuitka, or any command that writes `build-output`.
