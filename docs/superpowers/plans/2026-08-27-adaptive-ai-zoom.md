# Adaptive AI Zoom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add same-frame adaptive 1.5×/2.0× refinement for small distant AI Aim targets only while normal Makcu movement is gated, with mapped Overlay boxes and immediate 1.0× fallback.

**Architecture:** A new pure `ai_zoom.py` module owns factor selection, NumPy crop/resize, coordinate mapping, association, and immutable analysis composition. `AiService` keeps its full-field base pass, conditionally runs a second pass on the same frame, and publishes one generation-safe result; `JitterApp` provides a short-lock gate snapshot and renders generation-safe Zoom status without exposing Tk state to workers.

**Tech Stack:** Python 3.12, NumPy, ONNX Runtime DirectML, DXCam, Tkinter/ttk, `unittest`, existing Makcu services.

**Spec:** `docs/superpowers/specs/2026-08-27-adaptive-ai-zoom-design.md`

## Global Constraints

- Windows-only; preserve the fixed centered 320-by-320 DXCam capture and bundled `models/all_games_320.onnx` contract.
- Use only the approved ONNX Runtime DirectML, DXCam, and NumPy stack; add no dependency, model, profile, training path, OpenCV, Pillow, Torch, or Ultralytics.
- Keep every Tk widget and Tk variable access on the main thread.
- Share the adaptive zoom gate through one boolean snapshot under a short lock; the AI worker must not inspect Tk, `TriggerGate`, or Makcu service state.
- Adaptive refinement is eligible only for connected, Master-armed, AI-selected, normal Trigger/Modifier-active movement; never for Overlay-only viewing or Test 3s.
- Preserve generation checks before and after blocking capture/inference work and atomically publish target/frame pairs.
- STOP, gate release, disconnect, hotkey disable, AI disable, and shutdown must not wait for an inference interval before signaling movement cancellation and a false zoom gate.
- A refinement miss falls back to the same frame's base 1.0× result; the first refinement exception disables only refinement for that AI generation and logs once.
- Zoom status, crop transforms, gate state, and refined snapshots are runtime-only; keep configuration schema 5 unchanged.
- Do not run Nuitka unless the user separately requests a packaged build.
- Use TDD for every production change: write a focused failing test, observe the expected failure, implement minimally, then rerun the focused and full relevant suites.
- Preserve unrelated user changes and ignored `config.json`, backups, and `app.log`.

---

### Task 1: Pure zoom selection, crop, resize, and coordinate transform

**Files:**
- Create: `ai_zoom.py`
- Create: `tests/test_ai_zoom.py`
- Modify: `tests/test_distribution_metadata.py`
- Modify: `tests/test_entrypoints.py`

**Interfaces:**
- Consumes: `ai_targeting.Detection`, `ai_targeting.TargetSnapshot`; RGB uint8 NumPy frames shaped `(320, 320, 3)`.
- Produces:
  - `FRAME_SIZE: int = 320`
  - `ZoomTransform(left: int, top: int, size: int, factor: float)`
  - `select_zoom_factor(detection: Detection, target: TargetSnapshot | None) -> float`
  - `resize_rgb_bilinear(image: numpy.ndarray, output_size: int = 320) -> numpy.ndarray`
  - `build_zoom_input(frame: numpy.ndarray, target: TargetSnapshot, factor: float) -> tuple[numpy.ndarray, ZoomTransform]`
  - `map_detection(detection: Detection, transform: ZoomTransform) -> Detection | None`
  - `map_target(target: TargetSnapshot, transform: ZoomTransform) -> TargetSnapshot`

- [ ] **Step 1: Write failing factor-selection tests**

Create `tests/test_ai_zoom.py` with literal boundary cases. The production mutation these tests catch is using one class-independent threshold or zooming a target outside the approved center radius.

```python
import unittest

import numpy as np

from ai_targeting import Detection, TargetSnapshot
from ai_zoom import select_zoom_factor


class ZoomFactorTests(unittest.TestCase):
    def target(self, kind="head", x=160.0, y=160.0):
        return TargetSnapshot(1, 10.0, kind, x, y)

    def test_head_height_boundaries_select_exact_factors(self):
        cases = (
            (18.0, 2.0),
            (18.01, 1.5),
            (32.0, 1.5),
            (32.01, 1.0),
        )
        for height, expected in cases:
            with self.subTest(height=height):
                detection = Detection(150, 100, 170, 100 + height, 0.9, 7)
                self.assertEqual(
                    select_zoom_factor(detection, self.target()), expected
                )

    def test_player_height_boundaries_select_exact_factors(self):
        cases = (
            (64.0, 2.0),
            (64.01, 1.5),
            (112.0, 1.5),
            (112.01, 1.0),
        )
        for height, expected in cases:
            with self.subTest(height=height):
                detection = Detection(140, 80, 180, 80 + height, 0.9, 0)
                self.assertEqual(
                    select_zoom_factor(
                        detection, self.target("player")
                    ),
                    expected,
                )

    def test_missing_unsupported_or_outside_center_target_stays_one_x(self):
        head = Detection(150, 100, 170, 110, 0.9, 7)
        unsupported = Detection(150, 100, 170, 110, 0.9, 4)
        self.assertEqual(select_zoom_factor(head, None), 1.0)
        self.assertEqual(
            select_zoom_factor(unsupported, self.target()), 1.0
        )
        self.assertEqual(
            select_zoom_factor(head, self.target(x=256.01)), 1.0
        )
        self.assertEqual(
            select_zoom_factor(head, self.target(x=256.0)), 2.0
        )
```

- [ ] **Step 2: Run the new test to verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_ai_zoom.py -v
```

Expected: import error for missing `ai_zoom`, proving the new module and behavior do not exist.

- [ ] **Step 3: Implement immutable transform and factor selection minimally**

Create `ai_zoom.py` with these constants and definitions:

```python
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ai_targeting import Detection, TargetSnapshot


FRAME_SIZE = 320
MAX_CENTER_DISTANCE = 96.0
_HEAD_TWO_X_MAX = 18.0
_HEAD_ONE_HALF_X_MAX = 32.0
_PLAYER_TWO_X_MAX = 64.0
_PLAYER_ONE_HALF_X_MAX = 112.0
_ZOOM_SOURCE_SIZES = {1.5: 213, 2.0: 160}


@dataclass(frozen=True)
class ZoomTransform:
    left: int
    top: int
    size: int
    factor: float


def select_zoom_factor(
    detection: Detection,
    target: TargetSnapshot | None,
) -> float:
    if target is None:
        return 1.0
    if (
        math.hypot(target.aim_x - 160.0, target.aim_y - 160.0)
        > MAX_CENTER_DISTANCE
    ):
        return 1.0
    height = detection.y2 - detection.y1
    if detection.class_id == 7:
        if height <= _HEAD_TWO_X_MAX:
            return 2.0
        if height <= _HEAD_ONE_HALF_X_MAX:
            return 1.5
    elif detection.class_id == 0:
        if height <= _PLAYER_TWO_X_MAX:
            return 2.0
        if height <= _PLAYER_ONE_HALF_X_MAX:
            return 1.5
    return 1.0
```

- [ ] **Step 4: Verify factor tests pass**

Run the Task 1 focused command again. Expected: every factor-selection test
passes before any geometry contract is introduced.

- [ ] **Step 5: Add failing bilinear, crop-clamp, and mapping tests**

Extend the imports first:

```python
from ai_zoom import (
    ZoomTransform,
    build_zoom_input,
    map_detection,
    map_target,
    resize_rgb_bilinear,
    select_zoom_factor,
)
```

Then append the following behavior tests. They catch nearest-neighbor
substitution, wrong crop origins at edges, inconsistent point/box transforms,
and returning views owned by the capture buffer.

```python
class ZoomGeometryTests(unittest.TestCase):
    def test_bilinear_resize_has_hand_derived_center_and_owned_rgb_output(self):
        source = np.array(
            [
                [[0, 0, 0], [100, 100, 100]],
                [[200, 200, 200], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )
        resized = resize_rgb_bilinear(source, output_size=3)
        self.assertEqual(resized.shape, (3, 3, 3))
        self.assertEqual(resized.dtype, np.uint8)
        self.assertEqual(resized[1, 1].tolist(), [139, 139, 139])
        self.assertFalse(np.shares_memory(resized, source))

    def test_two_x_crop_clamps_at_top_left_and_maps_coordinates_back(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        target = TargetSnapshot(4, 20.0, "head", 20.0, 30.0)
        zoomed, transform = build_zoom_input(frame, target, 2.0)
        self.assertEqual(transform, ZoomTransform(0, 0, 160, 2.0))
        self.assertEqual(zoomed.shape, (320, 320, 3))
        self.assertEqual(zoomed.dtype, np.uint8)
        self.assertFalse(np.shares_memory(zoomed, frame))
        mapped = map_detection(
            Detection(20, 40, 100, 140, 0.8, 7), transform
        )
        self.assertEqual(
            mapped, Detection(10, 20, 50, 70, 0.8, 7)
        )
        self.assertEqual(
            map_target(
                TargetSnapshot(4, 20.0, "head", 80, 120), transform
            ),
            TargetSnapshot(4, 20.0, "head", 40, 60),
        )

    def test_one_half_x_crop_clamps_at_bottom_right(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        target = TargetSnapshot(5, 21.0, "player", 310.0, 300.0)
        _zoomed, transform = build_zoom_input(frame, target, 1.5)
        self.assertEqual(transform, ZoomTransform(107, 107, 213, 1.5))

    def test_crop_sizes_centering_and_all_four_edge_clamps(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        cases = (
            (2.0, 160.0, 160.0, ZoomTransform(80, 80, 160, 2.0)),
            (1.5, 160.0, 160.0, ZoomTransform(54, 54, 213, 1.5)),
            (2.0, 310.0, 20.0, ZoomTransform(160, 0, 160, 2.0)),
            (2.0, 20.0, 300.0, ZoomTransform(0, 160, 160, 2.0)),
            (2.0, 310.0, 300.0, ZoomTransform(160, 160, 160, 2.0)),
        )
        for factor, aim_x, aim_y, expected in cases:
            with self.subTest(factor=factor, aim_x=aim_x, aim_y=aim_y):
                target = TargetSnapshot(6, 22.0, "head", aim_x, aim_y)
                _zoomed, transform = build_zoom_input(frame, target, factor)
                self.assertEqual(transform, expected)

    def test_mapping_clamps_nonempty_box_to_source_bounds(self):
        transform = ZoomTransform(280, 280, 40, 8.0)
        self.assertEqual(
            map_detection(
                Detection(-100, -100, 400, 400, 0.8, 0), transform
            ),
            Detection(267.5, 267.5, 320.0, 320.0, 0.8, 0),
        )

    def test_mapping_discards_empty_box_after_clamping(self):
        transform = ZoomTransform(0, 0, 40, 8.0)
        self.assertIsNone(
            map_detection(
                Detection(-100, -100, -1, -1, 0.8, 0), transform
            )
        )
```

- [ ] **Step 6: Run Task 1 tests and verify the geometry assertions fail**

Run the Task 1 focused command. Expected: import error for the newly imported
geometry functions. Step 4 already recorded the green factor-only baseline,
so this RED result is attributable to the new geometry contract.

- [ ] **Step 7: Implement resize, crop, and mapping**

Implement `resize_rgb_bilinear` with `numpy.linspace`, floor/ceiling source
indices, four bilinear weights, clipping, half-up rounding via `+ 0.5`, and a
fresh contiguous uint8 result:

```python
def resize_rgb_bilinear(
    image: np.ndarray,
    output_size: int = FRAME_SIZE,
) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] < 1
        or image.shape[1] < 1
        or image.dtype != np.uint8
    ):
        raise ValueError("Resize source must be a nonempty RGB uint8 array")
    if isinstance(output_size, bool) or int(output_size) != output_size:
        raise ValueError("Output size must be a positive integer")
    output_size = int(output_size)
    if output_size < 1:
        raise ValueError("Output size must be a positive integer")

    source_height, source_width = image.shape[:2]
    source_x = np.linspace(0.0, source_width - 1, output_size)
    source_y = np.linspace(0.0, source_height - 1, output_size)
    x0 = np.floor(source_x).astype(np.intp)
    y0 = np.floor(source_y).astype(np.intp)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx = (source_x - x0)[None, :, None]
    wy = (source_y - y0)[:, None, None]
    top_left = image[y0[:, None], x0[None, :]].astype(np.float32)
    top_right = image[y0[:, None], x1[None, :]].astype(np.float32)
    bottom_left = image[y1[:, None], x0[None, :]].astype(np.float32)
    bottom_right = image[y1[:, None], x1[None, :]].astype(np.float32)
    blended = (
        top_left * (1.0 - wx) * (1.0 - wy)
        + top_right * wx * (1.0 - wy)
        + bottom_left * (1.0 - wx) * wy
        + bottom_right * wx * wy
    )
    rounded = np.floor(np.clip(blended, 0.0, 255.0) + 0.5)
    return np.ascontiguousarray(rounded.astype(np.uint8))
```

Implement `build_zoom_input` as:

```python
def build_zoom_input(
    frame: np.ndarray,
    target: TargetSnapshot,
    factor: float,
) -> tuple[np.ndarray, ZoomTransform]:
    if not isinstance(frame, np.ndarray) or frame.shape != (320, 320, 3):
        raise ValueError("Zoom source must be RGB 320x320x3")
    if frame.dtype != np.uint8:
        raise ValueError("Zoom source must use uint8 pixels")
    try:
        size = _ZOOM_SOURCE_SIZES[float(factor)]
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Zoom factor must be 1.5 or 2.0") from error
    left = max(0, min(320 - size, round(target.aim_x - size / 2)))
    top = max(0, min(320 - size, round(target.aim_y - size / 2)))
    transform = ZoomTransform(left, top, size, float(factor))
    crop = frame[top:top + size, left:left + size]
    return resize_rgb_bilinear(crop), transform
```

Implement mapping with one shared coordinate function:

```python
def _map_coordinate(value: float, origin: int, scale: float) -> float:
    return max(0.0, min(320.0, origin + value * scale))


def map_detection(
    detection: Detection,
    transform: ZoomTransform,
) -> Detection | None:
    scale = transform.size / 320.0
    x1 = _map_coordinate(detection.x1, transform.left, scale)
    y1 = _map_coordinate(detection.y1, transform.top, scale)
    x2 = _map_coordinate(detection.x2, transform.left, scale)
    y2 = _map_coordinate(detection.y2, transform.top, scale)
    if x2 <= x1 or y2 <= y1:
        return None
    return Detection(
        x1, y1, x2, y2, detection.confidence, detection.class_id
    )


def map_target(
    target: TargetSnapshot,
    transform: ZoomTransform,
) -> TargetSnapshot:
    scale = transform.size / 320.0
    return TargetSnapshot(
        target.sequence,
        target.captured_at,
        target.target_class,
        _map_coordinate(target.aim_x, transform.left, scale),
        _map_coordinate(target.aim_y, transform.top, scale),
    )
```

- [ ] **Step 8: Run Task 1 focused tests and the AI targeting suite**

Run:

```powershell
python -m unittest discover -s tests -p test_ai_zoom.py -v
python -m unittest discover -s tests -p test_ai_targeting.py -v
```

Expected: both commands pass with no warning or error output.

- [ ] **Step 9: Prove canonical source discovery sees the new module**

Before updating the expected sets, run:

```powershell
python -m unittest discover -s tests -p test_distribution_metadata.py -v
python -m unittest discover -s tests -p test_entrypoints.py -v
```

Expected: the exact compile-target assertions fail with `ai_zoom.py` present
in the actual set but absent from the expected set. This proves the canonical
source discovery already includes the new module and no parallel/manual source
list is needed.

- [ ] **Step 10: Update the literal expected compile inventories**

Add `"ai_zoom.py"` once beside the other AI modules in the expected
`compile_targets` sets in `tests/test_distribution_metadata.py` and
`tests/test_entrypoints.py`. Do not weaken either test to subset membership.

- [ ] **Step 11: Verify source inventory and review payload are green**

```powershell
python -m unittest discover -s tests -p test_distribution_metadata.py -v
python -m unittest discover -s tests -p test_entrypoints.py -v
python .\distribution_metadata.py --review-json
```

Expected: both suites pass; review JSON exits zero, contains `ai_zoom.py`
exactly once in `compile_targets`, and contains it once in the `py_compile`
argv.

- [ ] **Step 12: Commit Task 1**

```powershell
git add ai_zoom.py tests/test_ai_zoom.py tests/test_distribution_metadata.py tests/test_entrypoints.py
git commit -m "feat: add pure adaptive zoom geometry"
```

---

### Task 2: Refined-target association and Overlay-frame composition

**Files:**
- Modify: `ai_targeting.py`
- Modify: `ai_zoom.py`
- Modify: `tests/test_ai_targeting.py`
- Modify: `tests/test_ai_zoom.py`

**Interfaces:**
- Consumes Task 1: `ZoomTransform`, `map_detection`, immutable targeting records.
- Produces:
  - `ai_targeting.detection_aim_point(detection: Detection) -> tuple[str, float, float] | None`
  - `ai_zoom.compose_zoom_refinement(base: DetectionAnalysis, refined_detections: Iterable[Detection], transform: ZoomTransform, settings: AimSettings) -> DetectionAnalysis | None`

- [ ] **Step 1: Write a failing public aim-point contract test**

Add to `tests/test_ai_targeting.py`:

```python
from ai_targeting import detection_aim_point


class DetectionAimPointTests(unittest.TestCase):
    def test_public_aim_point_preserves_head_and_player_contract(self):
        self.assertEqual(
            detection_aim_point(Detection(10, 20, 30, 40, 0.9, 7)),
            ("head", 20.0, 30.0),
        )
        self.assertEqual(
            detection_aim_point(Detection(10, 20, 30, 120, 0.9, 0)),
            ("player", 20.0, 40.0),
        )
        self.assertIsNone(
            detection_aim_point(Detection(10, 20, 30, 40, 0.9, 4))
        )
```

- [ ] **Step 2: Run the targeting test and verify RED**

```powershell
python -m unittest discover -s tests -p test_ai_targeting.py -v
```

Expected: import error for `detection_aim_point`.

- [ ] **Step 3: Expose the existing pure aim-point helper**

Rename `_aim_point` to `detection_aim_point` in `ai_targeting.py` and update all
internal call sites. Do not change either class's aim-point formula.

- [ ] **Step 4: Verify targeting tests pass**

Run the Task 2 targeting command. Expected: PASS.

- [ ] **Step 5: Add failing refinement-composition tests**

Append to `tests/test_ai_zoom.py`. Use literal base and refined detections so
the expected index and mapped coordinates do not share production helpers.

```python
from ai_targeting import AimSettings, DetectionAnalysis, DetectionFrameSnapshot
from ai_zoom import compose_zoom_refinement


class ZoomCompositionTests(unittest.TestCase):
    def base_player(self):
        return DetectionAnalysis(
            TargetSnapshot(7, 30.0, "player", 160.0, 120.0),
            DetectionFrameSnapshot(
                7,
                30.0,
                (
                    Detection(20, 40, 60, 140, 0.7, 0),
                    Detection(140, 80, 180, 280, 0.8, 0),
                    Detection(250, 40, 300, 160, 0.75, 0),
                ),
                1,
            ),
        )

    def test_player_seed_refines_to_mapped_head_and_preserves_other_boxes(self):
        result = compose_zoom_refinement(
            self.base_player(),
            (Detection(140, 70, 180, 110, 0.92, 7),),
            ZoomTransform(80, 40, 160, 2.0),
            AimSettings(confidence=0.35),
        )
        self.assertIsNotNone(result)
        self.assertEqual(
            result.target,
            TargetSnapshot(7, 30.0, "head", 160.0, 85.0),
        )
        self.assertEqual(result.frame.selected_index, 1)
        self.assertEqual(result.frame.detections[0], self.base_player().frame.detections[0])
        self.assertEqual(
            result.frame.detections[1],
            Detection(150, 75, 170, 95, 0.92, 7),
        )
        self.assertEqual(result.frame.detections[2], self.base_player().frame.detections[2])

    def test_head_seed_rejects_player_downgrade(self):
        base = DetectionAnalysis(
            TargetSnapshot(8, 31.0, "head", 160.0, 100.0),
            DetectionFrameSnapshot(
                8, 31.0, (Detection(150, 90, 170, 110, 0.9, 7),), 0
            ),
        )
        self.assertIsNone(
            compose_zoom_refinement(
                base,
                (Detection(100, 80, 220, 300, 0.95, 0),),
                ZoomTransform(80, 40, 160, 2.0),
                AimSettings(confidence=0.35),
            )
        )

    def test_outside_expanded_seed_and_low_confidence_fall_back(self):
        for detection in (
            Detection(300, 10, 319, 29, 0.9, 7),
            Detection(140, 70, 180, 110, 0.1, 7),
        ):
            with self.subTest(detection=detection):
                self.assertIsNone(
                    compose_zoom_refinement(
                        self.base_player(),
                        (detection,),
                        ZoomTransform(80, 40, 160, 2.0),
                        AimSettings(confidence=0.35),
                    )
                )

    def test_association_includes_exact_12px_and_20_percent_margins(self):
        transform = ZoomTransform(80, 0, 160, 2.0)
        exact_boundary = Detection(86, 70, 106, 90, 0.92, 7)
        result = compose_zoom_refinement(
            self.base_player(),
            (exact_boundary,),
            transform,
            AimSettings(confidence=0.35),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.target.aim_x, 128.0)
        self.assertEqual(result.target.aim_y, 40.0)

        for just_outside in (
            Detection(85.96, 70, 105.96, 90, 0.92, 7),
            Detection(86, 69.96, 106, 89.96, 0.92, 7),
        ):
            with self.subTest(just_outside=just_outside):
                self.assertIsNone(
                    compose_zoom_refinement(
                        self.base_player(),
                        (just_outside,),
                        transform,
                        AimSettings(confidence=0.35),
                    )
                )
```

- [ ] **Step 6: Run zoom tests and verify RED**

Run the Task 1 zoom command. Expected: import error or assertion failures for
missing `compose_zoom_refinement`; Task 1 geometry remains green.

- [ ] **Step 7: Implement association and composition minimally**

Extend the `ai_zoom.py` imports with `Iterable`, `AimSettings`,
`DetectionAnalysis`, `DetectionFrameSnapshot`, `analyze_detections`, and
`detection_aim_point`, then implement:

```python
def compose_zoom_refinement(
    base: DetectionAnalysis,
    refined_detections: Iterable[Detection],
    transform: ZoomTransform,
    settings: AimSettings,
) -> DetectionAnalysis | None:
    selected_index = base.frame.selected_index
    if (
        base.target is None
        or selected_index is None
        or not 0 <= selected_index < len(base.frame.detections)
    ):
        return None
    seed = base.frame.detections[selected_index]
    if seed.class_id == 7:
        allowed_classes = {7}
    elif seed.class_id == 0:
        allowed_classes = {0, 7}
    else:
        return None

    margin_x = max(12.0, (seed.x2 - seed.x1) * 0.20)
    margin_y = max(12.0, (seed.y2 - seed.y1) * 0.20)
    compatible = []
    for detection in refined_detections:
        mapped = map_detection(detection, transform)
        if mapped is None or mapped.class_id not in allowed_classes:
            continue
        point = detection_aim_point(mapped)
        if point is None:
            continue
        _target_class, aim_x, aim_y = point
        if (
            seed.x1 - margin_x <= aim_x <= seed.x2 + margin_x
            and seed.y1 - margin_y <= aim_y <= seed.y2 + margin_y
        ):
            compatible.append(mapped)

    refined = analyze_detections(
        compatible,
        settings,
        sequence=base.frame.sequence,
        captured_at=base.frame.captured_at,
        previous=base.target,
    )
    if refined.target is None or refined.frame.selected_index is None:
        return None
    selected_refined = refined.frame.detections[
        refined.frame.selected_index
    ]
    composed = list(base.frame.detections)
    composed[selected_index] = selected_refined
    return DetectionAnalysis(
        refined.target,
        DetectionFrameSnapshot(
            base.frame.sequence,
            base.frame.captured_at,
            tuple(composed),
            selected_index,
        ),
    )
```

- [ ] **Step 8: Run focused zoom and targeting suites**

```powershell
python -m unittest discover -s tests -p test_ai_zoom.py -v
python -m unittest discover -s tests -p test_ai_targeting.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 2**

```powershell
git add ai_targeting.py ai_zoom.py tests/test_ai_targeting.py tests/test_ai_zoom.py
git commit -m "feat: compose associated zoom refinements"
```

---

### Task 3: Generation-safe dual-pass AI service

**Files:**
- Modify: `ai_service.py`
- Modify: `tests/test_ai_service.py`

**Interfaces:**
- Consumes Task 1: `build_zoom_input`, `select_zoom_factor`.
- Consumes Task 2: `compose_zoom_refinement`.
- Changes: `AiService.start(settings_provider, zoom_gate_provider=None) -> int | None`; omitted provider means refinement is always disabled.
- Produces: generation-safe `AiEvent("zoom", factor)` transitions with payload in `{1.0, 1.5, 2.0}`.

- [ ] **Step 1: Extend detector and capture fakes before writing service assertions**

Add `import numpy as np` to `tests/test_ai_service.py`, then add a sequential
detector that records exact frame objects and can run a callback during the
second inference:

```python
class SequentialDetector:
    provider = "DmlExecutionProvider"

    def __init__(self, outputs, second_call_hook=None):
        self.outputs = list(outputs)
        self.frames = []
        self.second_call_hook = second_call_hook

    def detect(self, frame):
        self.frames.append(frame)
        if len(self.frames) == 2 and self.second_call_hook is not None:
            self.second_call_hook()
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output
```

Add this helper inside `AiServiceTests`; the existing `FakeCapture` stays alive
after its one or more frames are consumed, matching the real worker lifecycle:

```python
def make_zoom_service(self, detector, *, frames=None, clock=time.perf_counter):
    source_frames = (
        list(frames)
        if frames is not None
        else [np.zeros((320, 320, 3), dtype=np.uint8)]
    )
    events = []
    service = AiService(
        events.append,
        detector_factory=lambda _path: detector,
        capture_factory=lambda: FakeCapture(source_frames),
        clock=clock,
    )
    self.addCleanup(service.close)
    return service, events
```

- [ ] **Step 2: Write failing one-pass and two-pass service tests**

Add tests with a base Player of height 80 (1.5×) and an associated refined
Head. Assertions must observe published service snapshots and real event
payloads, not only fake call counts.

```python
def test_zoom_gate_false_publishes_base_with_one_inference(self):
    detector = SequentialDetector(((Detection(140, 80, 180, 160, 0.9, 0),),))
    service, events = self.make_zoom_service(detector)
    generation = service.start(AimSettings, lambda: False)
    self.assertTrue(wait_until(
        lambda: len(detector.frames) == 1
        and service.latest_snapshot() is not None
    ))
    self.assertIsNotNone(generation)
    self.assertEqual(len(detector.frames), 1)
    self.assertEqual(service.latest_snapshot().target_class, "player")
    self.assertNotIn(AiEvent("zoom", 1.5), events)


def test_eligible_gate_true_refines_same_frame_and_emits_zoom(self):
    detector = SequentialDetector((
        (Detection(140, 80, 180, 160, 0.9, 0),),
        (Detection(144, 135, 174, 165, 0.92, 7),),
    ))
    service, events = self.make_zoom_service(detector)
    service.start(AimSettings, lambda: True)
    self.assertTrue(wait_until(
        lambda: len(detector.frames) == 2
        and service.latest_snapshot() is not None
        and AiEvent("zoom", 1.5) in events
    ))
    self.assertEqual(len(detector.frames), 2)
    self.assertEqual(detector.frames[1].shape, (320, 320, 3))
    self.assertEqual(service.latest_snapshot().target_class, "head")
    self.assertIn(AiEvent("zoom", 1.5), events)
    self.assertEqual(service.latest_detection_snapshot().selected_index, 0)


def test_ineligible_large_target_uses_one_inference(self):
    detector = SequentialDetector(
        ((Detection(140, 80, 180, 193, 0.9, 0),),)
    )
    service, events = self.make_zoom_service(detector)
    service.start(AimSettings, lambda: True)
    self.assertTrue(wait_until(
        lambda: len(detector.frames) == 1
        and service.latest_snapshot() is not None
    ))
    self.assertEqual(len(detector.frames), 1)
    self.assertFalse(any(event.kind == "zoom" for event in events))


def test_refinement_miss_publishes_same_frame_base_fallback(self):
    base = (Detection(140, 80, 180, 160, 0.9, 0),)
    detector = SequentialDetector((base, ()))
    service, events = self.make_zoom_service(detector)
    service.start(AimSettings, lambda: True)
    self.assertTrue(wait_until(
        lambda: len(detector.frames) == 2
        and service.latest_snapshot() is not None
    ))
    self.assertEqual(service.latest_snapshot().target_class, "player")
    self.assertEqual(service.latest_detection_snapshot().detections, base)
    self.assertFalse(any(event.kind == "zoom" for event in events))


def test_exact_small_head_threshold_runs_two_x_second_pass(self):
    detector = SequentialDetector((
        (Detection(151, 150, 169, 168, 0.9, 7),),
        (Detection(142, 142, 178, 178, 0.93, 7),),
    ))
    service, events = self.make_zoom_service(detector)
    service.start(AimSettings, lambda: True)
    self.assertTrue(wait_until(
        lambda: len(detector.frames) == 2
        and AiEvent("zoom", 2.0) in events
    ))
    self.assertEqual(service.latest_snapshot().target_class, "head")
    self.assertEqual(service.latest_detection_snapshot().selected_index, 0)
```

Use the repository's existing `wait_until`/thread-cleanup utilities and complete
fake structures rather than adding test-only methods to `AiService`.

- [ ] **Step 3: Run AI service tests and verify RED**

```powershell
python -m unittest discover -s tests -p test_ai_service.py -v
```

Expected: FAIL because `start` does not accept the zoom provider and no second
pass or zoom event exists.

- [ ] **Step 4: Implement the minimal dual-pass path**

Import the pure zoom functions into `ai_service.py`:

```python
from ai_zoom import (
    build_zoom_input,
    compose_zoom_refinement,
    select_zoom_factor,
)
```

Change `start` to normalize an omitted provider to a callable returning false,
pass it into `_worker`, and keep `refinement_enabled = True` plus
`published_factor = 1.0` local to the generation.

```python
def start(
    self,
    settings_provider: Callable[[], AimSettings],
    zoom_gate_provider: Callable[[], bool] | None = None,
) -> int | None:
    if zoom_gate_provider is None:
        zoom_gate_provider = lambda: False
    # Preserve the existing lifecycle body, but append zoom_gate_provider to
    # the worker thread's args tuple.


def _worker(
    self,
    generation: int,
    stop_event: threading.Event,
    settings_provider: Callable[[], AimSettings],
    zoom_gate_provider: Callable[[], bool],
) -> None:
    # Preserve existing setup; initialize these beside sequence/previous.
    refinement_enabled = True
    published_factor = 1.0
```

After the existing base analysis, execute this structure:

```python
factor = 1.0
published = base_analysis
if refinement_enabled and bool(zoom_gate_provider()):
    selected = base_analysis.frame.selected_index
    if selected is not None and base_analysis.target is not None:
        seed = base_analysis.frame.detections[selected]
        requested_factor = select_zoom_factor(seed, base_analysis.target)
        if requested_factor > 1.0 and bool(zoom_gate_provider()):
            if not self._is_current(generation, stop_event):
                return
            zoomed, transform = build_zoom_input(
                frame, base_analysis.target, requested_factor
            )
            if not self._is_current(generation, stop_event):
                return
            refined_detections = detector.detect(zoomed)
            if not self._is_current(generation, stop_event):
                return
            refined = compose_zoom_refinement(
                base_analysis,
                refined_detections,
                transform,
                settings,
            )
            if refined is not None:
                published = refined
                factor = requested_factor
```

Take one `settings = settings_provider()` snapshot per captured frame and use
it for both analyses. Recheck current generation immediately before the second
detector call and after it. Atomically assign `published.target` and
`published.frame` under the existing lock, then set
`previous = published.target` so the next base pass follows the coordinate
actually used for movement.

Emit `AiEvent("zoom", factor)` through `_emit_current` only when
`factor != published_factor`, then set `published_factor = factor`. Count one
completed inference frame after the single atomic publication, regardless of
detector call count.

- [ ] **Step 5: Verify success, miss, and gate-false tests pass**

Run the AI service suite. Expected: the new basic tests and all existing
generation/publication tests pass.

- [ ] **Step 6: Add failing gate-release and stale-generation tests**

Add one test whose second-call hook changes a mutable gate provider to false;
assert that the published result is the base Player, no 1.5× event is emitted,
and no refined Head is published. Add another using the existing stop/restart
interleaving helpers whose hook calls `service.stop("test")`; assert neither
target nor frame from that obsolete generation is published.

```python
def test_gate_release_during_second_call_discards_refinement(self):
    gate = {"active": True}
    detector = SequentialDetector(
        (
            (Detection(140, 80, 180, 160, 0.9, 0),),
            (Detection(144, 135, 174, 165, 0.95, 7),),
        ),
        second_call_hook=lambda: gate.update(active=False),
    )
    service, events = self.make_zoom_service(detector)
    service.start(AimSettings, lambda: gate["active"])
    self.assertTrue(wait_until(
        lambda: len(detector.frames) == 2
        and service.latest_snapshot() is not None
    ))
    self.assertEqual(service.latest_snapshot().target_class, "player")
    self.assertNotIn(AiEvent("zoom", 1.5), events)


def test_restart_during_second_call_cannot_publish_old_refinement(self):
    new_detection = Detection(30, 30, 40, 40, 0.9, 7)
    new_detector = SequentialDetector(((new_detection,),))
    service_holder = {}

    def restart_service():
        service_holder["service"].stop("restart")
        service_holder["service"].start(AimSettings, lambda: False)

    old_detector = SequentialDetector(
        (
            (Detection(140, 80, 180, 160, 0.9, 0),),
            (Detection(144, 135, 174, 165, 0.95, 7),),
        ),
        second_call_hook=restart_service,
    )
    detectors = iter((old_detector, new_detector))
    captures = iter((
        FakeCapture([np.zeros((320, 320, 3), dtype=np.uint8)]),
        FakeCapture([np.zeros((320, 320, 3), dtype=np.uint8)]),
    ))
    events = []
    service = AiService(
        events.append,
        detector_factory=lambda _path: next(detectors),
        capture_factory=lambda: next(captures),
    )
    service_holder["service"] = service
    self.addCleanup(service.close)

    service.start(AimSettings, lambda: True)
    self.assertTrue(wait_until(
        lambda: service.latest_detection_snapshot() is not None
        and service.latest_detection_snapshot().detections == (new_detection,)
    ))
    self.assertNotIn(AiEvent("zoom", 1.5), events)


def test_zoom_events_emit_only_on_success_and_factor_transition(self):
    base = (Detection(140, 80, 180, 160, 0.9, 0),)
    refined = (Detection(144, 135, 174, 165, 0.92, 7),)
    detector = SequentialDetector((base, refined, base, refined, base, ()))
    frames = [np.zeros((320, 320, 3), dtype=np.uint8) for _ in range(3)]
    service, events = self.make_zoom_service(detector, frames=frames)
    service.start(AimSettings, lambda: True)
    self.assertTrue(wait_until(
        lambda: len(detector.frames) == 6
        and service.latest_snapshot() is not None
        and service.latest_snapshot().sequence == 3
        and AiEvent("zoom", 1.0) in events
    ))
    self.assertEqual(
        [event.payload for event in events if event.kind == "zoom"],
        [1.5, 1.0],
    )
    self.assertEqual(service.latest_snapshot().target_class, "player")


def test_fps_counts_published_frames_not_detector_calls(self):
    base = (Detection(140, 80, 180, 160, 0.9, 0),)
    refined = (Detection(144, 135, 174, 165, 0.92, 7),)
    detector = SequentialDetector((base, refined, base, refined))
    frames = [np.zeros((320, 320, 3), dtype=np.uint8) for _ in range(2)]
    service, events = self.make_zoom_service(
        detector,
        frames=frames,
        clock=FakeClock([0.0, 0.1, 0.2, 0.3, 1.2]),
    )
    service.start(AimSettings, lambda: True)
    self.assertTrue(wait_until(
        lambda: any(event.kind == "fps" for event in events)
    ))
    fps = next(event.payload for event in events if event.kind == "fps")
    self.assertAlmostEqual(fps, 2 / 1.2)
    self.assertEqual(len(detector.frames), 4)
```

- [ ] **Step 7: Run AI service tests and verify the new interleaving fails**

Run the Task 3 service command. Expected: at least one new interleaving,
transition, or FPS assertion fails until post-call ownership and one-publication
accounting are complete.

- [ ] **Step 8: Implement post-call discard and generation ownership**

Replace the code immediately after the second detector call with the explicit
ownership checks:

```python
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
    if refined is not None:
        published = refined
        factor = requested_factor
```

When generation is obsolete, return without publishing. When only the gate is
false, the initialized base analysis and 1.0× factor remain selected. Never
hold `_lock` while calling a provider, NumPy, or ONNX Runtime.

- [ ] **Step 9: Add failing refinement-error containment test**

Use three captured frames: the first successfully publishes 1.5×, the second
raises during refinement and falls back to 1.0×, and the third proves base
inference continues without another second pass:

```python
def test_first_refinement_error_disables_zoom_once_for_generation(self):
    base = (Detection(140, 80, 180, 160, 0.9, 0),)
    refined = (Detection(144, 135, 174, 165, 0.92, 7),)
    detector = SequentialDetector(
        (base, refined, base, RuntimeError("refine failed"), base)
    )
    frames = [np.zeros((320, 320, 3), dtype=np.uint8) for _ in range(3)]
    service, events = self.make_zoom_service(detector, frames=frames)

    with self.assertLogs("ai_service", level="ERROR") as logs:
        service.start(AimSettings, lambda: True)
        self.assertTrue(wait_until(
            lambda: len(detector.frames) == 5
            and service.latest_snapshot() is not None
            and service.latest_snapshot().sequence == 3
            and AiEvent("zoom", 1.0) in events
        ))

    matching_logs = [
        line for line in logs.output
        if "Adaptive AI zoom disabled" in line
    ]
    self.assertEqual(len(matching_logs), 1)
    self.assertEqual(service.latest_snapshot().target_class, "player")
    self.assertFalse(any(event.kind == "error" for event in events))
    self.assertEqual(
        [event.payload for event in events if event.kind == "zoom"],
        [1.5, 1.0],
    )
    self.assertEqual(len(detector.frames), 5)
```

- [ ] **Step 10: Implement one-shot refinement disable**

Wrap only gate reads, zoom selection, crop/resize, second inference, mapping,
and composition in the refinement exception boundary. On the first exception:

```python
LOGGER.exception(
    "Adaptive AI zoom disabled for generation %s", generation
)
refinement_enabled = False
factor = 1.0
published = base_analysis
```

Do not catch the base detector call in this local boundary. The outer worker
exception handler must retain the existing fail-closed behavior.

- [ ] **Step 11: Run service, zoom, and targeting suites**

```powershell
python -m unittest discover -s tests -p test_ai_service.py -v
python -m unittest discover -s tests -p test_ai_zoom.py -v
python -m unittest discover -s tests -p test_ai_targeting.py -v
```

Expected: all pass with no leaked threads or unexpected logs.

- [ ] **Step 12: Commit Task 3**

```powershell
git add ai_service.py tests/test_ai_service.py
git commit -m "feat: run generation-safe adaptive zoom"
```

---

### Task 4: Thread-safe UI gate snapshot and Zoom status

**Files:**
- Modify: `ui.py`
- Modify: `tests/test_ui.py`

**Interfaces:**
- Consumes Task 3: `AiService.start(settings_provider, zoom_gate_provider)` and `AiEvent("zoom", float)`.
- Produces:
  - `JitterApp.get_adaptive_zoom_gate() -> bool`
  - private main-thread `_sync_adaptive_zoom_gate()`
  - `ai_zoom_var: tkinter.StringVar`, rendered as `1.0×`, `1.5×`, or `2.0×`.

- [ ] **Step 1: Extend the AI service stub with the complete new start contract**

In `tests/test_ui.py`, change `StubAiService.start` to accept and retain both
providers:

```python
def start(self, settings_provider, zoom_gate_provider=None):
    self.start_calls.append((settings_provider, zoom_gate_provider))
    if self.start_exception is not None:
        raise self.start_exception
    if not self.start_result:
        return self.start_result
    self.generation += 1
    self.active_generation = self.generation
    return self.active_generation
```

Existing assertions use only `len(start_calls)` or compare it with `[]`, so
they remain unchanged. The new provider test below unpacks the tuple, verifies
the bound getter identity, and calls it to assert the observable boolean value.

- [ ] **Step 2: Write failing gate-matrix tests**

Add behavior tests that exercise real `JitterApp` state transitions:

```python
def test_adaptive_zoom_gate_requires_connected_normal_ai_movement_gate(self):
    self.service.connected = True
    self.app.toggle_ai_source()
    self.app.set_master(True)
    self.assertFalse(self.app.get_adaptive_zoom_gate())
    _settings_provider, zoom_provider = self.ai.start_calls[-1]
    self.assertIs(zoom_provider.__self__, self.app)
    self.assertIs(
        zoom_provider.__func__, self.app.get_adaptive_zoom_gate.__func__
    )
    self.assertFalse(zoom_provider())

    self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertTrue(self.app.get_adaptive_zoom_gate())

    self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
    self.assertFalse(self.app.get_adaptive_zoom_gate())


def test_overlay_only_jitter_only_and_test_run_never_enable_zoom_gate(self):
    self.service.connected = True
    self.app.toggle_overlay()
    self.assertFalse(self.app.get_adaptive_zoom_gate())
    self.app.toggle_overlay()
    self.app.toggle_jitter_source()
    self.app.set_master(True)
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertFalse(self.app.get_adaptive_zoom_gate())
    self.app.emergency_stop()
    self.app.ai_selected = True
    self.app.start_test_run()
    self.assertFalse(self.app.get_adaptive_zoom_gate())


def test_configured_modifier_requires_both_buttons_for_zoom_gate(self):
    self.app.modifier_var.set("Right")
    self.app.on_bindings_changed()
    self.prepare_armed_sources(MotionSources(False, True))
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertFalse(self.app.get_adaptive_zoom_gate())
    self.app.handle_service_event(ServiceEvent("button", ("Right", True)))
    self.assertTrue(self.app.get_adaptive_zoom_gate())
    self.app.handle_service_event(ServiceEvent("button", ("Right", False)))
    self.assertFalse(self.app.get_adaptive_zoom_gate())


def test_source_removal_and_hotkey_disable_clear_zoom_gate(self):
    self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
    self.assertTrue(self.app.get_adaptive_zoom_gate())
    self.app.toggle_ai_source()
    self.assertFalse(self.app.get_adaptive_zoom_gate())

    self.app.close_app()
    self.make_app()
    self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
    self.app._cancel_after("_ui_pump_after_id")
    self.app._hotkey_pressed()
    self.drain_ui_queue()
    self.assertFalse(self.app.get_adaptive_zoom_gate())


def test_disconnect_clears_zoom_gate(self):
    self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
    self.app.handle_service_event(ServiceEvent("disconnected", "lost"))
    self.assertFalse(self.app.get_adaptive_zoom_gate())


def test_ai_error_clears_zoom_gate(self):
    self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
    with self.assertLogs(level="ERROR"):
        self.app.handle_ai_event(AiEvent("error", "failed"))
    self.assertFalse(self.app.get_adaptive_zoom_gate())


def test_stop_clears_zoom_gate(self):
    self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
    self.app.emergency_stop("Stopped")
    self.assertFalse(self.app.get_adaptive_zoom_gate())


def test_close_clears_zoom_gate(self):
    self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
    self.app.close_app()
    self.assertFalse(self.app.get_adaptive_zoom_gate())
```

Each terminal action must expose false before the test allows a queued worker
observation.

- [ ] **Step 3: Run UI tests and verify RED**

```powershell
python -m unittest discover -s tests -p test_ui.py -v
```

Expected: FAIL because the getter and synchronized snapshot do not exist and
the service receives no zoom provider.

- [ ] **Step 4: Implement the short-lock snapshot**

Initialize `_adaptive_zoom_gate = False` beside `_ai_snapshot` under
`_ai_lock`. Implement:

```python
def get_adaptive_zoom_gate(self) -> bool:
    with self._ai_lock:
        return self._adaptive_zoom_gate

def _sync_adaptive_zoom_gate(self) -> None:
    active = bool(
        not self._closing
        and self.service.connected
        and self._ai_runtime_active
        and self.master_armed
        and self.ai_selected
        and self._motion_mode is None
        and self.trigger_gate.active
    )
    with self._ai_lock:
        self._adaptive_zoom_gate = active
    if not active:
        self.ai_zoom_var.set("1.0×")
```

Call `_sync_adaptive_zoom_gate()` immediately after button-state updates and
before potentially blocking motion/runtime operations in Master disable, source
change, Test entry/restore, STOP, disconnect, AI error, and shutdown paths.
Also call it after binding changes clear held state and after every successful
Master/source/AI-runtime enabling transition. Pass
`self.get_adaptive_zoom_gate` as the second provider in `_start_ai_runtime`.
Every call site is on the Tk thread; worker reads invoke only the getter.

- [ ] **Step 5: Run the gate-matrix tests and fix missed transitions**

Run the UI suite. Expected: all gate tests pass. If an existing transition
test exposes a missed false state, add the sync call to that exact transition;
do not make the worker derive UI state.

- [ ] **Step 6: Write failing Zoom metric tests**

Add UI assertions for the fourth runtime metric and generation-owned events:

```python
def test_zoom_metric_starts_one_x_and_tracks_valid_events(self):
    self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")
    self.assertIn("ZOOM", widget_texts(self.app))
    self.app.handle_ai_event(AiEvent("zoom", 2.0))
    self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")
    self.app._ai_runtime_active = True
    self.app.handle_ai_event(AiEvent("zoom", 1.5))
    self.assertEqual(self.app.ai_zoom_var.get(), "1.5×")
    self.app.handle_ai_event(AiEvent("zoom", 2.0))
    self.assertEqual(self.app.ai_zoom_var.get(), "2.0×")
    self.app.handle_ai_event(AiEvent("zoom", "invalid"))
    self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")


def test_stop_disconnect_and_ai_stop_reset_zoom_metric(self):
    self.app.ai_zoom_var.set("2.0×")
    self.app.emergency_stop()
    self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")
    self.app.ai_zoom_var.set("1.5×")
    self.app.handle_service_event(ServiceEvent("disconnected", "lost"))
    self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")
    self.app._ai_runtime_active = True
    self.app.ai_zoom_var.set("2.0×")
    self.app.handle_ai_event(AiEvent("stopped", "manual"))
    self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")


def test_trigger_release_resets_zoom_metric_without_waiting_for_ai_frame(self):
    self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
    self.app.handle_ai_event(AiEvent("zoom", 1.5))
    self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
    self.assertFalse(self.app.get_adaptive_zoom_gate())
    self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")
```

Extend stale UI epoch coverage so a queued `zoom` event from an obsolete AI
epoch cannot change `ai_zoom_var`:

```python
def test_stale_queued_zoom_event_cannot_change_metric_after_stop(self):
    self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
    self.app._cancel_after("_ui_pump_after_id")
    self.app.queue_ai_event(AiEvent("zoom", 2.0))
    self.app.emergency_stop("Stopped")
    self.drain_ui_queue()
    self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")
```

- [ ] **Step 7: Run UI tests and verify metric failures**

Run the Task 4 UI command. Expected: FAIL because no ZOOM metric or event
branch exists.

- [ ] **Step 8: Implement the metric and event handling**

Create `ai_zoom_var = tk.StringVar(value="1.0×")` with the other AI runtime
variables. Add `("ZOOM", self.ai_zoom_var)` to the AI RUNTIME metric sequence;
move Overlay/Color/Head button grid rows down without changing the fixed outer
geometry or removing scrolling.

Accept `zoom` only while the current AI runtime is active. Normalize payloads
to the literal set `{1.0, 1.5, 2.0}` and display one decimal followed by `×`;
invalid/non-finite payloads display `1.0×`. Reset the variable in loading,
runtime stop, fallback error, STOP, disconnect, and shutdown paths. A false
result from `_sync_adaptive_zoom_gate()` also resets it immediately, so Trigger
or Modifier release does not wait for another captured frame. Include `zoom`
in the `handle_ai_event` active-runtime guard; `queue_ai_event` already stamps
every AI event with `_ai_event_epoch`, so no second generation mechanism is
added.

Use this literal event branch after the existing FPS branch:

```python
elif kind == "zoom":
    try:
        factor = float(event.payload)
        if not math.isfinite(factor) or factor not in {1.0, 1.5, 2.0}:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        factor = 1.0
    self.ai_zoom_var.set(f"{factor:.1f}×")
```

- [ ] **Step 9: Run UI, service, and Overlay suites**

```powershell
python -m unittest discover -s tests -p test_ui.py -v
python -m unittest discover -s tests -p test_ai_service.py -v
python -m unittest discover -s tests -p test_overlay.py -v
```

Expected: all pass; Tk geometry and lifecycle tests remain green.

- [ ] **Step 10: Commit Task 4**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: gate and display adaptive AI zoom"
```

---

### Task 5: User and repository documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes Tasks 1-4: implemented runtime behavior and the Task 1 canonical
  compile/review inventory.
- Produces: user and repository guidance matching the implementation.

- [ ] **Step 1: Confirm the canonical inventory baseline is already green**

```powershell
python -m unittest discover -s tests -p test_distribution_metadata.py -v
python -m unittest discover -s tests -p test_entrypoints.py -v
python .\distribution_metadata.py --review-json
```

Expected: both suites and review JSON pass, with `ai_zoom.py` exactly once in
`compile_targets` and once in the generated `py_compile` argv. If this baseline
is not green, return to Task 1 and fix the inventory before editing docs.

- [ ] **Step 2: Update README and AGENTS guidance**

Document these exact user-visible rules:

- Adaptive Zoom is automatic and has no persisted control.
- It refines a small target that the 1.0× base pass already selected; it does
  not replace full-field target acquisition.
- It runs a second pass only during connected, Master-armed, AI-selected,
  Trigger/Modifier-active normal movement.
- `ZOOM` reports 1.0×, 1.5×, or 2.0×.
- Same-frame base fallback, mapped Overlay boxes, Overlay-only exclusion, and
  Test 3s exclusion.
- Schema 5 remains unchanged and zoom status is runtime-only.
- `ai_zoom.py` owns pure zoom geometry/refinement.
- Verification compile commands include `ai_zoom.py`.

Do not claim that display magnification improves inference and do not describe
adaptive zoom as active while idle.

- [ ] **Step 3: Run documentation-adjacent tests and review JSON**

```powershell
python -m unittest discover -s tests -p test_distribution_metadata.py -v
python -m unittest discover -s tests -p test_entrypoints.py -v
python .\distribution_metadata.py --review-json
```

Expected: tests pass; JSON exits zero and lists `ai_zoom.py` once.

- [ ] **Step 4: Commit Task 5**

```powershell
git add README.md AGENTS.md
git commit -m "docs: document adaptive AI zoom"
```

---

### Task 6: Complete automated verification

**Files:**
- Verify only; do not edit `build-output/`, `dist/`, generated caches, or user data.

**Interfaces:**
- Consumes: the completed Task 1-5 branch.
- Produces: fresh command evidence for compile, tests, imports, DirectML model/provider contract, and distribution plan.

- [ ] **Step 1: Compile every source module**

```powershell
python -m py_compile main.py ui.py motion.py combined_motion.py ai_targeting.py ai_zoom.py ai_detection.py ai_capture.py ai_service.py overlay.py makcu_service.py hotkeys.py settings.py sound_service.py liquid_widgets.py distribution_metadata.py
```

Expected: exit code 0 with no traceback.

- [ ] **Step 2: Run the complete test suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass with zero failures/errors and no leaked-worker warning.

- [ ] **Step 3: Verify runtime imports**

```powershell
python -c "import makcu, serial, onnxruntime, dxcam, comtypes, numpy"
```

Expected: exit code 0.

- [ ] **Step 4: Verify the real model hash and DirectML provider**

```powershell
python .\main.py --ai-runtime-self-check
```

Expected JSON: `"status": "ok"`, matching bundled model SHA-256, and
`"provider": "DmlExecutionProvider"`.

- [ ] **Step 5: Verify distribution review and diff hygiene**

```powershell
python .\distribution_metadata.py --review-json
git diff --check
git status --short --branch
```

Expected: review exits zero and contains `ai_zoom.py`; diff check exits zero;
working tree is clean on the current implementation branch. Do not run Nuitka.

- [ ] **Step 6: Record verification outcome without a source commit**

Report exact test count, exit codes, provider, and any skipped hardware items.
This is a verify-only task; do not create an empty commit.

---

### Task 7: Live Makcu and Overlay acceptance

**Files:**
- Verify only against the source launcher; do not package.

**Interfaces:**
- Consumes: green Task 6 source tree, connected Makcu, a game scene with near and distant detectable targets.
- Produces: physical acceptance evidence and observed log timestamps.

- [ ] **Step 1: Launch the exact source tree and establish a clean log boundary**

Close any older Jitter process through its window, record the current
`app.log` line count, and run:

```powershell
.\run_gui.bat
```

Expected: the window title is `Jitter — Makcu Control`, Makcu connects, AI
runtime reports DirectML when selected, and Zoom starts at `1.0×`.

- [ ] **Step 2: Verify zoom eligibility and levels**

With AI Aim selected and Master armed:

- leave Trigger released and confirm `1.0×`;
- hold Trigger/Modifier on a near target and confirm `1.0×`;
- hold on a medium distant target and confirm `1.5×`;
- hold on a very small target and confirm `2.0×`;
- release either required button and confirm immediate `1.0×`.

Record whether each level appears and the corresponding target size/scene.

- [ ] **Step 3: Verify mapped Overlay composition**

Enable Overlay and confirm:

- unrelated base boxes remain visible during 1.5×/2.0× refinement;
- the selected refined box remains aligned to the real target rather than the
  resized crop coordinates;
- selected width, configured color, and Head Boxes filter remain correct;
- a refinement miss returns to the base box without blanking the Overlay;
- DXCam capture does not contain Overlay rectangles.

- [ ] **Step 4: Verify safety transitions with physical movement**

For AI-only and combined Jitter+AI movement, verify Trigger release, Modifier
release, Mouse5 disable, STOP, disconnect, reconnect, and window close. Each
must prevent further Makcu reports according to the existing barrier contract;
Zoom must reset to `1.0×`; no stale refined movement may appear after restart.

- [ ] **Step 5: Verify excluded modes and logs**

Confirm Overlay-only viewing and every Test 3s source matrix remain at `1.0×`.
Inspect only log lines after the Step 1 boundary for adaptive-zoom exceptions,
AI errors, stale generation symptoms, or Overlay failures.

- [ ] **Step 6: Report acceptance result**

List each physical check as pass/fail with the observed Zoom state and relevant
new `app.log` lines. If a check fails, reproduce it, add a failing automated
test where possible, apply TDD, rerun Task 6, and repeat only the affected live
check plus STOP/disconnect regression checks.
