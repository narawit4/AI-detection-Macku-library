# Recoil-Stable Adaptive Zoom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Adaptive Zoom resist recoil and Jitter by confirming base-pass target stability before AI movement and by holding 2.0x refinement at 1.5x until a 100 ms recoil cooldown expires.

**Architecture:** `ai_zoom.py` gains an immutable pure state machine that observes only base-pass targets and returns confirmation and factor-limit decisions. Each `AiService` worker owns fresh stability and selection-continuity state, publishes current-frame Overlay detections independently from the movement target, and retains the existing two-inference maximum and generation barriers.

**Tech Stack:** Python 3.12, immutable dataclasses, NumPy, ONNX Runtime DirectML, DXCam, Tkinter/ttk, `unittest`, existing Makcu and combined-motion services.

**Spec:** `docs/superpowers/specs/2026-08-27-recoil-stable-adaptive-zoom-design.md`

## Global Constraints

- Preserve the fixed centered 320-by-320 capture, bundled `models/all_games_320.onnx`, DirectML provider preference, existing crop geometry, and existing optimized NumPy resize.
- Add no 3.0x mode, third inference, model, dependency, UI control, configuration field, profile, training path, prediction, or stale-target hold.
- The stability state is immutable, pure, hardware-free, Tk-free, local to one AI worker generation, and never persisted.
- Stability observes base-pass targets only. Refined coordinates never become its previous target.
- Keep selection continuity separate from movement publication: an unconfirmed current target may guide the next base association but must not be exposed through `latest_snapshot()`.
- A false zoom gate resets only stability policy. It must not change the existing one-pass Overlay selection behavior.
- Preserve generation checks before and after every blocking detector call and atomically publish the movement target and detection frame under the existing lock.
- Preserve immediate STOP, Trigger/Modifier release, hotkey disable, disconnect, AI stop, restart, and shutdown cancellation. Stability must not delay the Makcu cancellation path.
- Combined Jitter+AI keeps Jitter active when the AI snapshot is `None`; use the existing pure combined-motion contract rather than adding a second mover.
- Keep at most two detector calls for each captured frame. A downgraded 2.0x request runs one 1.5x refinement, not both.
- Do not run Nuitka. Preserve ignored `config.json`, `config.json.bak`, and `app.log` as user data.
- Use TDD for production changes: add focused tests, observe the expected failure, implement minimally, then run the focused and complete relevant suites.

## File and Interface Map

- `ai_zoom.py` owns `ZoomStabilityState`, fixed constants, observation, refinement-miss, movement-confirmation, and applied-factor decisions alongside the existing zoom geometry.
- `ai_service.py` owns per-generation `selection_previous` and `ZoomStabilityState`, orders gate reads and inference, and separates `DetectionAnalysis.target` from `DetectionAnalysis.frame` during confirmation.
- `tests/test_ai_zoom.py` locks exact 18.0/18.01 and 100 ms boundaries plus immutable transition behavior.
- `tests/test_ai_service.py` locks publication, factor transitions, selection continuity, refinement fallback, detector-call counts, and generation reset.
- `tests/test_combined_motion.py` already proves that a `None` AI target contributes zero while Jitter continues; it is a required regression suite but needs no production change.
- `README.md` and `AGENTS.md` document the runtime behavior and fixed policy without presenting it as a user setting.

---

### Task 1: Add the pure recoil-stability state machine

**Files:**
- Modify: `ai_zoom.py`
- Modify: `tests/test_ai_zoom.py`

**Interfaces:**
- Consumes: consecutive base-pass `TargetSnapshot | None`, injected monotonic `now: float`, and requested factors from existing `select_zoom_factor`.
- Produces:
  - `STABLE_DISPLACEMENT_PX: float = 18.0`
  - `STABLE_CONFIRMATION_COUNT: int = 2`
  - `RECOIL_COOLDOWN_SECONDS: float = 0.100`
  - `ZoomStabilityState(previous_base_target=None, stable_count=0, cooldown_until=0.0)`
  - `observe_zoom_stability(state, target, now) -> ZoomStabilityState`
  - `record_zoom_refinement_miss(state, now) -> ZoomStabilityState`
  - `movement_is_confirmed(state) -> bool`
  - `limit_zoom_factor(requested_factor, state, now) -> float`

- [ ] **Step 1: Add failing immutable-state and observation tests**

Extend the imports in `tests/test_ai_zoom.py` exactly as follows:

```python
from dataclasses import FrozenInstanceError

from ai_zoom import (
    RECOIL_COOLDOWN_SECONDS,
    ZoomStabilityState,
    limit_zoom_factor,
    movement_is_confirmed,
    observe_zoom_stability,
    record_zoom_refinement_miss,
)
```

Add this test class before `ZoomGeometryTests`:

```python
class ZoomStabilityTests(unittest.TestCase):
    def target(self, sequence=1, kind="head", x=100.0, y=100.0):
        return TargetSnapshot(sequence, 10.0, kind, x, y)

    def test_initial_state_is_empty_unconfirmed_and_immutable(self):
        state = ZoomStabilityState()
        self.assertIsNone(state.previous_base_target)
        self.assertEqual(state.stable_count, 0)
        self.assertEqual(state.cooldown_until, 0.0)
        self.assertFalse(movement_is_confirmed(state))
        with self.assertRaises(FrozenInstanceError):
            state.stable_count = 1

    def test_first_acquisition_starts_count_and_cooldown(self):
        target = self.target()
        state = observe_zoom_stability(
            ZoomStabilityState(), target, 10.0
        )
        self.assertEqual(state.previous_base_target, target)
        self.assertEqual(state.stable_count, 1)
        self.assertAlmostEqual(
            state.cooldown_until, 10.0 + RECOIL_COOLDOWN_SECONDS
        )
        self.assertFalse(movement_is_confirmed(state))

    def test_exact_18_pixels_confirms_but_18_point_01_restarts(self):
        first = observe_zoom_stability(
            ZoomStabilityState(), self.target(x=100.0), 10.0
        )
        exact = observe_zoom_stability(
            first, self.target(sequence=2, x=118.0), 10.01
        )
        outside = observe_zoom_stability(
            first, self.target(sequence=2, x=118.01), 10.01
        )
        self.assertEqual(exact.stable_count, 2)
        self.assertTrue(movement_is_confirmed(exact))
        self.assertEqual(outside.stable_count, 1)
        self.assertFalse(movement_is_confirmed(outside))
        self.assertAlmostEqual(outside.cooldown_until, 10.11)

    def test_class_change_is_unstable_inside_distance_boundary(self):
        first = observe_zoom_stability(
            ZoomStabilityState(), self.target(kind="player"), 10.0
        )
        changed = observe_zoom_stability(
            first,
            self.target(sequence=2, kind="head", x=101.0),
            10.02,
        )
        self.assertEqual(changed.stable_count, 1)
        self.assertEqual(changed.previous_base_target.target_class, "head")
        self.assertAlmostEqual(changed.cooldown_until, 10.12)

    def test_missing_target_clears_previous_and_confirmation(self):
        first = observe_zoom_stability(
            ZoomStabilityState(), self.target(), 10.0
        )
        confirmed = observe_zoom_stability(
            first, self.target(sequence=2), 10.01
        )
        missing = observe_zoom_stability(confirmed, None, 10.02)
        self.assertIsNone(missing.previous_base_target)
        self.assertEqual(missing.stable_count, 0)
        self.assertFalse(movement_is_confirmed(missing))
        self.assertAlmostEqual(missing.cooldown_until, 10.12)

    def test_two_x_requires_confirmation_and_exact_cooldown_boundary(self):
        first = observe_zoom_stability(
            ZoomStabilityState(), self.target(), 10.0
        )
        confirmed = observe_zoom_stability(
            first, self.target(sequence=2), 10.02
        )
        self.assertEqual(limit_zoom_factor(1.0, first, 10.0), 1.0)
        self.assertEqual(limit_zoom_factor(1.5, first, 10.0), 1.5)
        self.assertEqual(limit_zoom_factor(2.0, first, 10.2), 1.5)
        self.assertEqual(limit_zoom_factor(2.0, confirmed, 10.099), 1.5)
        self.assertEqual(limit_zoom_factor(2.0, confirmed, 10.1), 2.0)

    def test_refinement_miss_resets_count_keeps_target_and_extends_only(self):
        target = self.target()
        state = ZoomStabilityState(target, 2, 20.0)
        retained = record_zoom_refinement_miss(state, 10.0)
        extended = record_zoom_refinement_miss(retained, 20.0)
        self.assertEqual(retained.previous_base_target, target)
        self.assertEqual(retained.stable_count, 0)
        self.assertEqual(retained.cooldown_until, 20.0)
        self.assertEqual(extended.previous_base_target, target)
        self.assertEqual(extended.stable_count, 0)
        self.assertAlmostEqual(extended.cooldown_until, 20.1)
```

- [ ] **Step 2: Run the pure zoom suite and verify RED**

```powershell
python -m unittest discover -s tests -p test_ai_zoom.py -v
```

Expected: import errors for the new stability interface. Existing factor,
resize, mapping, and composition tests must remain otherwise unchanged.

- [ ] **Step 3: Implement the immutable state machine minimally**

Add these constants and functions in `ai_zoom.py` after the existing zoom
threshold constants and before `ZoomTransform`:

```python
STABLE_DISPLACEMENT_PX = 18.0
STABLE_CONFIRMATION_COUNT = 2
RECOIL_COOLDOWN_SECONDS = 0.100


@dataclass(frozen=True)
class ZoomStabilityState:
    previous_base_target: TargetSnapshot | None = None
    stable_count: int = 0
    cooldown_until: float = 0.0


def _extended_zoom_cooldown(
    state: ZoomStabilityState,
    now: float,
) -> float:
    return max(
        state.cooldown_until,
        now + RECOIL_COOLDOWN_SECONDS,
    )


def observe_zoom_stability(
    state: ZoomStabilityState,
    target: TargetSnapshot | None,
    now: float,
) -> ZoomStabilityState:
    if target is None:
        return ZoomStabilityState(
            None,
            0,
            _extended_zoom_cooldown(state, now),
        )

    previous = state.previous_base_target
    unstable = (
        previous is None
        or previous.target_class != target.target_class
        or math.hypot(
            target.aim_x - previous.aim_x,
            target.aim_y - previous.aim_y,
        ) > STABLE_DISPLACEMENT_PX
    )
    if unstable:
        return ZoomStabilityState(
            target,
            1,
            _extended_zoom_cooldown(state, now),
        )
    return ZoomStabilityState(
        target,
        min(STABLE_CONFIRMATION_COUNT, state.stable_count + 1),
        state.cooldown_until,
    )


def record_zoom_refinement_miss(
    state: ZoomStabilityState,
    now: float,
) -> ZoomStabilityState:
    return ZoomStabilityState(
        state.previous_base_target,
        0,
        _extended_zoom_cooldown(state, now),
    )


def movement_is_confirmed(state: ZoomStabilityState) -> bool:
    return (
        state.previous_base_target is not None
        and state.stable_count >= STABLE_CONFIRMATION_COUNT
    )


def limit_zoom_factor(
    requested_factor: float,
    state: ZoomStabilityState,
    now: float,
) -> float:
    if requested_factor == 2.0 and (
        not movement_is_confirmed(state)
        or now < state.cooldown_until
    ):
        return 1.5
    return float(requested_factor)
```

Do not add schema normalization or mutable setters. These helpers receive only
the factors already produced by `select_zoom_factor`.

- [ ] **Step 4: Run focused and adjacent pure suites**

```powershell
python -m unittest discover -s tests -p test_ai_zoom.py -v
python -m unittest discover -s tests -p test_ai_targeting.py -v
```

Expected: all tests pass. The exact 18.0 case confirms, 18.01 restarts, and
2.0x becomes eligible at `now == cooldown_until`.

- [ ] **Step 5: Review and commit the pure interface**

```powershell
git diff --check
git diff -- ai_zoom.py tests/test_ai_zoom.py
git add ai_zoom.py tests/test_ai_zoom.py
git commit -m "feat: add recoil-stable zoom state"
```

Verify the diff contains no service, UI, configuration, model, dependency, or
packaging change.

---

### Task 2: Integrate confirmation and cooldown into the AI worker

**Files:**
- Modify: `ai_service.py`
- Modify: `tests/test_ai_service.py`
- Verify: `tests/test_combined_motion.py`

**Interfaces:**
- Consumes Task 1's pure state functions and the existing
  `zoom_gate_provider: Callable[[], bool]`.
- Maintains two distinct per-generation values:
  - `selection_previous: TargetSnapshot | None` for existing 48-pixel base
    association, updated only from `base_analysis.target`;
  - `stability: ZoomStabilityState` for base displacement, confirmation, and
    cooldown, reset whenever a worker observes a false gate.
- Produces one atomic `DetectionAnalysis` publication per captured frame. Its
  current frame remains available to the Overlay while its movement target is
  replaced with `None` until confirmed.
- Keeps `AiEvent("zoom", factor)` transition behavior, but emits the applied
  successful factor after stability limiting.

- [ ] **Step 1: Add a deterministic frame and clock harness**

Add these helpers beside the existing fake capture and clock classes in
`tests/test_ai_service.py`:

```python
class ControlledCapture(FakeCapture):
    def __init__(self, frames):
        super().__init__(frames)
        self._permits = threading.Semaphore(0)

    def release_frame(self):
        self._permits.release()

    def read(self):
        if not self.frames:
            return None
        if not self._permits.acquire(timeout=0.01):
            return None
        return super().read()


class MutableClock:
    def __init__(self, value=0.0):
        self._value = value
        self._lock = threading.Lock()

    def set(self, value):
        with self._lock:
            self._value = value

    def __call__(self):
        with self._lock:
            return self._value
```

Change `AiServiceTests.make_zoom_service` and add two small test helpers:

```python
def make_zoom_service(
    self,
    detector,
    *,
    frames=None,
    capture=None,
    clock=time.perf_counter,
):
    source_frames = (
        list(frames)
        if frames is not None
        else [np.zeros((320, 320, 3), dtype=np.uint8)]
    )
    if capture is None:
        capture = FakeCapture(source_frames)
    events = []
    service = AiService(
        events.append,
        detector_factory=lambda _path: detector,
        capture_factory=lambda: capture,
        clock=clock,
    )
    self.addCleanup(service.close)
    return service, events

def small_head(self, x=160.0):
    return Detection(x - 9.0, 150.0, x + 9.0, 168.0, 0.9, 7)

def release_and_wait(self, capture, service, sequence):
    capture.release_frame()
    self.assertTrue(wait_until(
        lambda: service.latest_detection_snapshot() is not None
        and service.latest_detection_snapshot().sequence == sequence
    ))
```

- [ ] **Step 2: Add failing acquisition, cooldown, recoil, and call-count tests**

Add these tests to `AiServiceTests`:

```python
def test_first_small_target_uses_one_half_x_but_withholds_movement(self):
    base = (self.small_head(),)
    refined = (Detection(142, 142, 178, 178, 0.93, 7),)
    detector = SequentialDetector((base, refined))
    capture = ControlledCapture(
        [np.zeros((320, 320, 3), dtype=np.uint8)]
    )
    clock = MutableClock(10.0)
    service, events = self.make_zoom_service(
        detector, capture=capture, clock=clock
    )
    service.start(AimSettings, lambda: True)

    self.release_and_wait(capture, service, 1)

    self.assertIsNone(service.latest_snapshot())
    self.assertEqual(service.latest_detection_snapshot().selected_index, 0)
    self.assertEqual(
        [event.payload for event in events if event.kind == "zoom"],
        [1.5],
    )
    self.assertEqual(len(detector.frames), 2)

def test_stable_target_enters_two_x_then_recoil_downgrades(self):
    base = (self.small_head(),)
    jumped = (self.small_head(178.01),)
    refined = (Detection(142, 142, 178, 178, 0.93, 7),)
    detector = SequentialDetector((
        base, refined,
        base, refined,
        base, refined,
        jumped, refined,
    ))
    capture = ControlledCapture([
        np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(4)
    ])
    clock = MutableClock(10.0)
    service, events = self.make_zoom_service(
        detector, capture=capture, clock=clock
    )
    service.start(AimSettings, lambda: True)

    self.release_and_wait(capture, service, 1)
    self.assertIsNone(service.latest_snapshot())
    self.assertEqual(len(detector.frames), 2)

    clock.set(10.05)
    self.release_and_wait(capture, service, 2)
    self.assertIsNotNone(service.latest_snapshot())
    self.assertEqual(len(detector.frames), 4)

    clock.set(10.1)
    self.release_and_wait(capture, service, 3)
    self.assertIsNotNone(service.latest_snapshot())
    self.assertEqual(len(detector.frames), 6)

    clock.set(10.11)
    self.release_and_wait(capture, service, 4)
    self.assertIsNone(service.latest_snapshot())
    self.assertEqual(len(detector.frames), 8)
    self.assertEqual(
        [event.payload for event in events if event.kind == "zoom"],
        [1.5, 2.0, 1.5],
    )
```

This one test proves the first 2.0x request is capped, movement resumes on the
second stable base frame before cooldown, the exact 100 ms boundary enables
2.0x, an 18.01-pixel jump immediately withholds movement, and every frame uses
exactly two detector calls.

- [ ] **Step 3: Add failing refinement-miss and honest Overlay fallback test**

```python
def test_refinement_miss_resets_confirmation_without_holding_old_target(self):
    base = (self.small_head(),)
    refined = (Detection(142, 142, 178, 178, 0.93, 7),)
    detector = SequentialDetector((
        base, refined,
        base, refined,
        base, (),
        base, refined,
        base, refined,
    ))
    capture = ControlledCapture([
        np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(5)
    ])
    clock = MutableClock(10.0)
    service, events = self.make_zoom_service(
        detector, capture=capture, clock=clock
    )
    service.start(AimSettings, lambda: True)

    self.release_and_wait(capture, service, 1)
    clock.set(10.05)
    self.release_and_wait(capture, service, 2)
    self.assertIsNotNone(service.latest_snapshot())

    clock.set(10.1)
    self.release_and_wait(capture, service, 3)
    self.assertIsNone(service.latest_snapshot())
    self.assertEqual(service.latest_detection_snapshot().detections, base)

    clock.set(10.11)
    self.release_and_wait(capture, service, 4)
    self.assertIsNone(service.latest_snapshot())

    clock.set(10.12)
    self.release_and_wait(capture, service, 5)
    self.assertIsNotNone(service.latest_snapshot())
    self.assertEqual(len(detector.frames), 10)
    self.assertEqual(
        [event.payload for event in events if event.kind == "zoom"],
        [1.5, 1.0, 1.5],
    )
```

The fifth frame confirms movement can resume after two stable observations
while the miss-created cooldown still prevents 2.0x.

- [ ] **Step 4: Add failing gate reset and selection-continuity test**

```python
def test_false_gate_resets_stability_but_preserves_base_selection(self):
    associated = Detection(80, 80, 120, 160, 0.9, 0)
    centered = Detection(140, 80, 180, 160, 0.9, 0)
    detector = SequentialDetector((
        (associated,), (),
        (associated, centered),
        (associated, centered), (),
    ))
    capture = ControlledCapture([
        np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(3)
    ])
    gate = {"active": True}
    clock = MutableClock(10.0)
    service, _events = self.make_zoom_service(
        detector, capture=capture, clock=clock
    )
    service.start(AimSettings, lambda: gate["active"])

    self.release_and_wait(capture, service, 1)
    self.assertIsNone(service.latest_snapshot())

    gate["active"] = False
    clock.set(10.01)
    self.release_and_wait(capture, service, 2)
    self.assertAlmostEqual(service.latest_snapshot().aim_x, 100.0)
    self.assertEqual(service.latest_detection_snapshot().selected_index, 0)

    gate["active"] = True
    clock.set(10.02)
    self.release_and_wait(capture, service, 3)
    self.assertIsNone(service.latest_snapshot())
    self.assertEqual(service.latest_detection_snapshot().selected_index, 0)
    self.assertEqual(len(detector.frames), 5)
```

If confirmation and selection share one previous target, the false-gate frame
selects the centered candidate instead and this test fails.

- [ ] **Step 5: Add failing worker-generation reset test**

```python
def test_restart_uses_fresh_stability_state(self):
    base = (self.small_head(),)
    refined = (Detection(142, 142, 178, 178, 0.93, 7),)
    old_detector = SequentialDetector((base, refined, base, refined))
    new_detector = SequentialDetector((base, refined))
    old_capture = ControlledCapture([
        np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(2)
    ])
    new_capture = ControlledCapture(
        [np.zeros((320, 320, 3), dtype=np.uint8)]
    )
    detectors = iter((old_detector, new_detector))
    captures = iter((old_capture, new_capture))
    clock = MutableClock(10.0)
    service = AiService(
        lambda _event: None,
        detector_factory=lambda _path: next(detectors),
        capture_factory=lambda: next(captures),
        clock=clock,
    )
    self.addCleanup(service.close)

    service.start(AimSettings, lambda: True)
    self.release_and_wait(old_capture, service, 1)
    clock.set(10.05)
    self.release_and_wait(old_capture, service, 2)
    self.assertIsNotNone(service.latest_snapshot())

    service.stop("restart")
    self.assertIsNone(service.latest_snapshot())
    clock.set(20.0)
    service.start(AimSettings, lambda: True)
    self.release_and_wait(new_capture, service, 1)
    self.assertIsNone(service.latest_snapshot())
    self.assertEqual(len(new_detector.frames), 2)
```

- [ ] **Step 6: Update existing service expectations to the approved contract**

Make these exact semantic changes to the existing tests before running RED:

- Remove `test_eligible_gate_true_refines_same_frame_and_emits_zoom`; Step 2's
  controlled first-acquisition test replaces it and additionally proves target
  suppression.
- In `test_ineligible_large_target_uses_one_inference`, wait for
  `latest_detection_snapshot()` instead of a non-`None` target, assert the
  target is `None`, assert selected index zero, and retain the one-call/no-zoom
  assertions. Confirmation applies even when the requested factor is 1.0x.
- In `test_refinement_miss_publishes_same_frame_base_fallback`, wait for the
  detection snapshot, assert `latest_snapshot()` is `None`, and retain the
  exact same-frame base detection tuple and no-zoom assertions.
- Remove `test_exact_small_head_threshold_runs_two_x_second_pass`; Steps 2 and
  3 lock the new first-frame cap and later exact cooldown entry.
- In `test_zoom_events_emit_only_on_success_and_factor_transition`, wait for
  detection sequence three instead of target sequence three and assert the
  final target is `None`; retain expected zoom transitions `[1.5, 1.0]`.
- Keep the gate-release-during-second-call test: the worker's final false gate
  observation must reset stability and publish the same-frame base target.
- Keep the restart-during-second-call, refinement-exception, FPS, atomic
  publication, stop, close, and stale-generation tests unchanged except for
  formatting forced by the helper signature.

- [ ] **Step 7: Run the service suite and verify RED**

```powershell
python -m unittest discover -s tests -p test_ai_service.py -v
```

Expected: failures show the current worker publishes movement on first
acquisition, immediately runs 2.0x, holds the old target after a miss, and has
no stability reset. Lifecycle and base one-pass tests should remain green.

- [ ] **Step 8: Import the pure interface into the service**

Add `DetectionAnalysis` to the existing `ai_targeting` import and extend the
`ai_zoom` import exactly as follows:

```python
from ai_targeting import (
    AimSettings,
    DetectionAnalysis,
    DetectionFrameSnapshot,
    TargetSnapshot,
    analyze_detections,
)
from ai_zoom import (
    ZoomStabilityState,
    build_zoom_input,
    compose_zoom_refinement,
    limit_zoom_factor,
    movement_is_confirmed,
    observe_zoom_stability,
    record_zoom_refinement_miss,
    select_zoom_factor,
)
```

- [ ] **Step 9: Replace the worker's shared previous-target flow**

At worker initialization, replace `previous = None` with:

```python
selection_previous = None
stability = ZoomStabilityState()
```

Replace the frame analysis, refinement, and publication block from
the `base_analysis = analyze_detections(` call through the old
`previous = published.target` assignment with this exact ordering:

```python
base_analysis = analyze_detections(
    detector.detect(frame),
    settings,
    sequence=sequence,
    captured_at=captured_at,
    previous=selection_previous,
)
if not self._is_current(generation, stop_event):
    return
selection_previous = base_analysis.target
factor = 1.0
published = base_analysis
gate_active = False
if refinement_enabled:
    try:
        gate_active = bool(zoom_gate_provider())
        if not gate_active:
            stability = ZoomStabilityState()
        else:
            stability = observe_zoom_stability(
                stability,
                base_analysis.target,
                captured_at,
            )
            selected = base_analysis.frame.selected_index
            if selected is not None and base_analysis.target is not None:
                seed = base_analysis.frame.detections[selected]
                requested_factor = select_zoom_factor(
                    seed,
                    base_analysis.target,
                )
                applied_factor = limit_zoom_factor(
                    requested_factor,
                    stability,
                    captured_at,
                )
                if applied_factor > 1.0:
                    if not bool(zoom_gate_provider()):
                        gate_active = False
                        stability = ZoomStabilityState()
                    else:
                        if not self._is_current(generation, stop_event):
                            return
                        zoomed, transform = build_zoom_input(
                            frame,
                            base_analysis.target,
                            applied_factor,
                        )
                        if not self._is_current(generation, stop_event):
                            return
                        refined_detections = detector.detect(zoomed)
                        if not self._is_current(generation, stop_event):
                            return
                        if bool(zoom_gate_provider()):
                            refined = compose_zoom_refinement(
                                base_analysis,
                                refined_detections,
                                transform,
                                settings,
                            )
                            if refined is None:
                                stability = record_zoom_refinement_miss(
                                    stability,
                                    self._clock(),
                                )
                            else:
                                published = refined
                                factor = applied_factor
                        else:
                            gate_active = False
                            stability = ZoomStabilityState()
            if gate_active and not movement_is_confirmed(stability):
                published = DetectionAnalysis(None, published.frame)
    except Exception:
        LOGGER.exception(
            "Adaptive AI zoom disabled for generation %s",
            generation,
        )
        refinement_enabled = False
        stability = ZoomStabilityState()
        factor = 1.0
        published = base_analysis
with self._lock:
    if not self._is_current_locked(generation, stop_event):
        return
    self._latest = published.target
    self._latest_detection = published.frame
```

Keep the existing factor-transition and FPS blocks immediately after this
publication. Do not assign refined or intentionally suppressed targets back to
`selection_previous`.

- [ ] **Step 10: Run focused worker and combined-motion suites**

```powershell
python -m unittest discover -s tests -p test_ai_service.py -v
python -m unittest discover -s tests -p test_combined_motion.py -v
```

Expected: all tests pass. The existing
`test_jitter_continues_when_ai_has_no_target` is the explicit proof that
confirmation suppresses only the AI component in combined mode.

- [ ] **Step 11: Run all AI, Overlay, UI, and Makcu regressions**

```powershell
python -m unittest discover -s tests -p "test_ai_*.py" -v
python -m unittest discover -s tests -p test_overlay.py -v
python -m unittest discover -s tests -p test_ui.py -v
python -m unittest discover -s tests -p test_makcu_service.py -v
```

Expected: no target/frame atomicity regression, no stale-generation
publication, no Overlay filtering change, no UI zoom-metric change, and no
Makcu gate or combined-source change.

- [ ] **Step 12: Review and commit worker integration**

```powershell
git diff --check
git diff -- ai_service.py tests/test_ai_service.py
git add ai_service.py tests/test_ai_service.py
git commit -m "feat: stabilize adaptive zoom during recoil"
```

Review the worker block against the spec in order: base analysis, selection
update, gate reset/observation, one limited refinement, miss handling,
confirmation suppression, atomic publish, factor event. Confirm there is no
third `detector.detect` call and no stale target hold.

---

### Task 3: Document the fixed recoil-stability policy

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Documents runtime-only behavior; exposes no new public Python API, UI field,
  configuration key, or packaging resource.

- [ ] **Step 1: Extend the README Adaptive Zoom section**

After the paragraph that explains same-frame fallback, add this exact text:

```markdown
While the movement gate is active, AI Aim confirms a base-pass target across
two consecutive same-class observations no more than 18 pixels apart. The
current Overlay boxes remain visible during the confirmation frame, but AI
movement is withheld rather than reusing an old position. In combined mode,
Jitter continues while that AI component is withheld.

A new or shaken small target starts with the wider 1.5x refinement. A confirmed
target may return to 2.0x only after the fixed 100 ms recoil cooldown. A normal
refinement miss also restarts confirmation and cooldown while preserving only
the current frame's base boxes. These constants are internal runtime policy,
not saved settings.
```

- [ ] **Step 2: Extend the AGENTS.md scope rules**

Immediately after the existing Adaptive Zoom bullets, add:

```markdown
- Recoil-stable zoom observes only base-pass targets. AI movement requires two
  consecutive same-class observations within 18 pixels; current Overlay boxes
  remain publishable while an unconfirmed movement target is `None`.
- A requested 2.0x refinement is capped at 1.5x until confirmation and a fixed
  100 ms cooldown both pass. A normal refinement miss resets confirmation and
  extends cooldown without adding an inference call or holding a stale target.
- Keep base-selection continuity separate from movement publication. Stability
  is local to one AI generation and resets when the movement zoom gate is
  false; combined Jitter continues when AI movement is unconfirmed.
```

- [ ] **Step 3: Verify documentation consistency**

```powershell
rg -n "18 pixels|100 ms|two consecutive|stale target|Jitter continues" README.md AGENTS.md
rg -n "3\.0x|third inference|persisted setting" README.md AGENTS.md
git diff --check
```

Expected: the first command finds the new policy in both files. The second
must not claim that 3.0x, a third inference, or a persisted stability setting
exists; any match must be a prohibition rather than a feature claim.

- [ ] **Step 4: Commit the documentation**

```powershell
git add README.md AGENTS.md
git commit -m "docs: explain recoil-stable adaptive zoom"
```

---

### Task 4: Run complete automated verification

**Files:**
- Verify only; do not edit generated output, caches, model, configuration, or
  log files.

**Interfaces:**
- Consumes the completed Task 1-3 source tree.
- Produces fresh evidence for syntax, every unit/integration-style test,
  dependency imports, the real DirectML model contract, and distribution
  review.

- [ ] **Step 1: Compile every source module**

```powershell
python -m py_compile main.py ui.py motion.py combined_motion.py ai_targeting.py ai_zoom.py ai_detection.py ai_capture.py ai_service.py overlay.py makcu_service.py hotkeys.py settings.py sound_service.py liquid_widgets.py distribution_metadata.py
```

Expected: exit code zero and no traceback.

- [ ] **Step 2: Run the complete test suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: every test passes with zero failures/errors and no worker leak.
Record the exact test count.

- [ ] **Step 3: Verify pinned runtime imports**

```powershell
python -c "import makcu, serial, onnxruntime, dxcam, comtypes, numpy"
```

Expected: exit code zero.

- [ ] **Step 4: Verify the real bundled model and DirectML provider**

```powershell
python .\main.py --ai-runtime-self-check
```

Expected JSON contains `"status": "ok"`, the approved model hash, and
`"provider": "DmlExecutionProvider"`.

- [ ] **Step 5: Verify distribution metadata and repository hygiene**

```powershell
python .\distribution_metadata.py --review-json
git diff --check
git status --short --branch
```

Expected: review exits zero, `ai_zoom.py` remains in the canonical source plan,
diff check exits zero, and the worktree is clean. Do not run `gen.bat` or
Nuitka.

- [ ] **Step 6: Record the verification result without an empty commit**

Report compile/import/review exit codes, exact test count, DirectML provider,
and any hardware checks still pending. This task is verify-only.

---

### Task 5: Run connected Makcu recoil acceptance

**Files:**
- Verify only with `run_gui.bat`, the connected Makcu, a detectable game scene,
  and new `app.log` lines after a recorded boundary. Do not package.

**Interfaces:**
- Consumes the green Task 4 source tree and real Trigger/Modifier/Mouse5 input.
- Produces physical acceptance evidence for AI-only and Jitter+AI modes plus a
  post-boundary log review.

- [ ] **Step 1: Launch the exact updated source tree**

Close any older Jitter window normally, record the current `app.log` line count
without changing the file, and run:

```powershell
.\run_gui.bat
```

Expected: the dashboard opens, Makcu connects, AI reports
`DmlExecutionProvider`, Overlay starts in its saved state, and `ZOOM` starts at
`1.0x`.

- [ ] **Step 2: Verify stable acquisition without firing**

Select AI Aim, arm Master, enable Overlay, and hold the configured
Trigger/Modifier over a small detectable target:

- the first acquisition uses `1.5x`, keeps current Overlay boxes, and produces
  no visible snap toward an old position;
- AI movement resumes on the next stable same-class observation;
- `2.0x` appears only after at least 100 ms of stability;
- releasing a required button immediately returns `ZOOM` to `1.0x` and stops
  movement.

Record pass/fail for each observation.

- [ ] **Step 3: Verify recoil behavior in AI-only mode**

Begin from a stable `2.0x` target and fire long enough to move the base target
more than the stability boundary:

- `ZOOM` drops to `1.5x` on the shaken acquisition rather than attempting a
  narrow 2.0x crop;
- AI movement pauses for the unconfirmed acquisition frame and never snaps to
  the previous target;
- Overlay boxes follow only current frames and never hold an old box through a
  miss;
- after two stable observations AI movement resumes, while `2.0x` waits for
  the cooldown.

- [ ] **Step 4: Verify recoil behavior in combined Jitter+AI mode**

Select both sources with the user's saved Jitter settings and repeat Step 3.
Confirm Jitter motion continues during the AI confirmation frame, the AI
component does not jump to a stale target, final reports remain clamped, and no
excess movement is queued for a later tick.

- [ ] **Step 5: Verify fallback and safety transitions**

Force or observe a normal refinement miss and confirm the current base boxes
remain visible, movement is withheld until reconfirmed, and the next 2.0x
request remains at 1.5x during cooldown. Then check Trigger release, configured
Modifier release if present, Mouse5 disable, STOP, disconnect, reconnect, AI
restart, and window close. Each must retain immediate cancellation and no stale
movement after restart.

- [ ] **Step 6: Review only new log lines and report acceptance**

Inspect `app.log` only after the Step 1 boundary for stability, refinement,
generation, DirectML, Overlay, and Makcu errors. Report every physical check as
pass/fail with the observed zoom transition. If a check fails, reproduce it,
add the smallest failing automated test, apply TDD, rerun Task 4, and repeat the
affected live check plus STOP/disconnect regression checks.
