# Full Primary-Display Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Jitter's centered 320-by-320 physical capture with one
aspect-preserving inference over the complete primary display while retaining
correct screen-space targeting, Adaptive Zoom, Overlay alignment, and the
fixed 1,000 Hz motion servo.

**Architecture:** DXCam will publish native primary-display RGB frames. The
detector will letterbox each base frame or zoom crop into the active square
model, decode both output contracts in model space, and apply one shared
inverse transform into the detector call's source space. Immutable snapshots
will carry native frame dimensions through targeting, movement, service, and
Overlay projection; zoom alone will use a temporary logical 320-square policy
space for its existing thresholds.

**Tech Stack:** Python 3, NumPy, ONNX Runtime DirectML, DXCam, Tkinter,
`unittest`, existing Makcu integration.

**Spec:** `docs/superpowers/specs/2026-08-31-full-primary-display-detection-design.md`

## Global Constraints

- Detect exactly the complete primary display (`output_idx=0`), not the
  virtual desktop or other monitors.
- Run exactly one base inference for every processed capture frame; never add
  tiles, overlapping regions, rotating scans, or queued catch-up work.
- Preserve aspect ratio by letterboxing into exactly 160, 320, or 640 square
  model input; fill padding with RGB value 114 and never stretch content.
- Use deterministic positive half-up dimension rounding
  `floor(value + 0.5)`; place an odd extra padding pixel on the right or
  bottom.
- Derive inverse scales from actual integer resized dimensions, not the
  theoretical floating-point gain.
- Keep the legacy `[1,300,6]` and raw one-class `[1,5,K]` ONNX contracts,
  DirectML-first provider policy, metadata validation, and runtime-only model
  selection unchanged.
- Keep the bundled `models/all_games_320.onnx` as the only bundled and
  packaged model. Never add, copy, persist, download, or package an external
  model or its path.
- Keep no more than one eligible Adaptive Zoom refinement call after the base
  call. Overlay-only inference, idle operation, and `Test 3s` never refine.
- Preserve detector order as the exact-distance selection tie break and never
  add target history, holds, recovery delays, or identity preference.
- Preserve the fixed 1,000 Hz motion servo, absolute deadlines, missed-slot
  skipping, immediate cancellation, zero-delta suppression, Max Step,
  acceleration, fractional carry, 150 ms freshness, and excess discard.
- Do not change the configuration schema, dashboard controls, dependencies,
  model packaging, or Nuitka plan.
- Keep all Tk access on the main thread and all blocking capture/inference work
  off the Tk event loop.
- Follow strict RED-GREEN-REFACTOR: every production behavior starts with a
  focused test that is observed failing for the intended reason.
- Preserve the user's untracked external `.onnx` files and never stage them.

## Agent Coordination

Use up to three subagents, but keep a single writing implementer at a time in
the shared feature worktree. The other seats may perform read-only next-task
analysis and independent task review. This prevents `targeting.py`, `zoom.py`,
`service.py`, and their positional snapshot constructors from being edited
concurrently while still overlapping analysis and review work.

---

### Task 1: Deterministic Rectangular RGB Resize

**Files:**
- Modify: `jitter_app/ai/resize.py`
- Test: `tests/test_image_resize.py`

**Interfaces:**
- Consumes: the current square
  `resize_rgb_bilinear(image: np.ndarray, output_size: int = 320)` behavior and
  regression hashes.
- Produces:
  `resize_rgb_bilinear_to(image: np.ndarray, output_width: int,
  output_height: int) -> np.ndarray` and a backward-compatible square wrapper.
- Produces a cached `_resize_plan(source_height, source_width, output_height,
  output_width=None)` whose three-argument form remains valid for existing
  cache tests.

- [ ] **Step 1: Add failing rectangular resize tests**

Add imports and tests with hand-derived expectations:

```python
from jitter_app.ai.resize import resize_rgb_bilinear_to

def test_rectangular_resize_has_exact_shape_pixels_and_ownership(self):
    source = np.array(
        [
            [[0, 0, 0], [100, 100, 100]],
            [[200, 200, 200], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    resized = resize_rgb_bilinear_to(source, 3, 2)

    self.assertEqual(resized.shape, (2, 3, 3))
    self.assertEqual(
        resized[:, :, 0].tolist(),
        [[0, 50, 100], [200, 228, 255]],
    )
    self.assertEqual(resized.dtype, np.uint8)
    self.assertTrue(resized.flags.c_contiguous)
    self.assertTrue(resized.flags.owndata)
    self.assertFalse(np.shares_memory(resized, source))

    vertical = resize_rgb_bilinear_to(source, 2, 3)
    self.assertEqual(
        vertical[:, :, 0].tolist(),
        [[0, 100], [100, 178], [200, 255]],
    )

def test_rectangular_resize_rejects_each_invalid_dimension(self):
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    for width, height in ((True, 2), (2, False), (0, 2), (2, 0),
                          (-1, 2), (2, -1), (1.5, 2), (2, 1.5)):
        with self.subTest(width=width, height=height):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                resize_rgb_bilinear_to(source, width, height)

def test_rectangular_plan_cache_keys_both_output_dimensions(self):
    image_resize._resize_plan.cache_clear()
    first = image_resize._resize_plan(2, 2, 2, 3)
    second = image_resize._resize_plan(2, 2, 2, 3)
    different = image_resize._resize_plan(2, 2, 3, 2)
    self.assertIs(first, second)
    self.assertIsNot(first, different)
```

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```powershell
python -m unittest tests.test_image_resize -v
```

Expected: import failure because `resize_rgb_bilinear_to` does not exist.

- [ ] **Step 3: Implement rectangular planning and resizing minimally**

Refactor the validation and cached plan so x coordinates use `output_width`
and y coordinates use `output_height`. Preserve the current float precision,
half-up pixel-value rounding, optimized separable paths, and immutable cached
arrays. Add this exact public boundary:

```python
def resize_rgb_bilinear_to(
    image: np.ndarray,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    _validate_source(image)
    output_width = _positive_output_dimension(output_width)
    output_height = _positive_output_dimension(output_height)
    source_height, source_width = image.shape[:2]
    x0, x1, y0, y1, wx, wy = _resize_plan(
        source_height,
        source_width,
        output_height,
        output_width,
    )
    return _blend_rgb(image, x0, x1, y0, y1, wx, wy)


def resize_rgb_bilinear(
    image: np.ndarray,
    output_size: int = 320,
) -> np.ndarray:
    return resize_rgb_bilinear_to(image, output_size, output_size)
```

The internal helpers must return the same `np.floor(np.clip(...)+0.5)` result
as the current implementation. Keep the existing 160/213-to-320 regression
hashes unchanged.

- [ ] **Step 4: Run RED-to-GREEN verification**

Run:

```powershell
python -m unittest tests.test_image_resize tests.test_ai_zoom tests.test_ai_detection -v
```

Expected: all resize, current zoom, and current detector tests pass; both
existing regression hashes remain exact.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- jitter_app/ai/resize.py tests/test_image_resize.py
git commit -m "feat: add deterministic rectangular rgb resize"
```

---

### Task 2: Shared Letterbox Detector Boundary

**Files:**
- Modify: `jitter_app/ai/detection.py`
- Modify: `jitter_app/ai/yolo.py`
- Test: `tests/test_ai_detection.py`
- Test: `tests/test_ai_yolo.py`

**Interfaces:**
- Consumes: `resize_rgb_bilinear_to(image, output_width, output_height)` from
  Task 1.
- Produces immutable `LetterboxTransform` and
  `build_letterbox_transform(source_width, source_height, input_size)`.
- Keeps `preprocess_frame(frame, input_size=320) -> np.ndarray` returning only
  contiguous normalized NCHW for callers/tests; an internal preparation path
  returns the identical transform used for that tensor.
- Changes `parse_output(output, input_size=320)` and
  `decode_single_class_yolo(output, input_size)` to return model-input-space
  boxes clipped only to `0..input_size`.
- Produces
  `map_detection_to_source(detection, transform) -> Detection | None`.
- Changes `OnnxDetector.detect(frame)` to accept any non-empty RGB `uint8`
  source frame and return detections in that call's source coordinates.

- [ ] **Step 1: Add failing transform and preprocessing tests**

Add these hand-checked fixtures:

```python
from jitter_app.ai.detection import (
    build_letterbox_transform,
    map_detection_to_source,
)

def test_letterbox_plans_landscape_portrait_and_half_up_odd_padding(self):
    landscape = build_letterbox_transform(1920, 1080, 320)
    portrait = build_letterbox_transform(1080, 1920, 320)
    odd = build_letterbox_transform(640, 361, 320)
    one_pixel = build_letterbox_transform(100000, 1, 160)
    ultrawide = build_letterbox_transform(3440, 1440, 320)
    square = build_letterbox_transform(777, 777, 160)

    self.assertEqual(
        (landscape.resized_width, landscape.resized_height,
         landscape.pad_left, landscape.pad_top,
         landscape.pad_right, landscape.pad_bottom),
        (320, 180, 0, 70, 0, 70),
    )
    self.assertEqual(
        (portrait.resized_width, portrait.resized_height,
         portrait.pad_left, portrait.pad_top,
         portrait.pad_right, portrait.pad_bottom),
        (180, 320, 70, 0, 70, 0),
    )
    self.assertEqual(
        (odd.resized_width, odd.resized_height,
         odd.pad_left, odd.pad_top, odd.pad_right, odd.pad_bottom),
        (320, 181, 0, 69, 0, 70),
    )
    self.assertEqual(
        (one_pixel.resized_width, one_pixel.resized_height,
         one_pixel.pad_left, one_pixel.pad_top,
         one_pixel.pad_right, one_pixel.pad_bottom),
        (160, 1, 0, 79, 0, 80),
    )
    self.assertEqual(
        (ultrawide.resized_width, ultrawide.resized_height,
         ultrawide.pad_left, ultrawide.pad_top,
         ultrawide.pad_right, ultrawide.pad_bottom),
        (320, 134, 0, 93, 0, 93),
    )
    self.assertEqual(
        (square.resized_width, square.resized_height,
         square.pad_left, square.pad_top,
         square.pad_right, square.pad_bottom),
        (160, 160, 0, 0, 0, 0),
    )

def test_preprocess_letterboxes_with_exact_114_padding(self):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, :, 0] = 255

    tensor = preprocess_frame(frame, 320)

    self.assertEqual(tensor.shape, (1, 3, 320, 320))
    np.testing.assert_allclose(tensor[0, :, 0, 0], [114 / 255] * 3)
    np.testing.assert_allclose(tensor[0, :, 70, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(tensor[0, :, 249, 319], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(tensor[0, :, 250, 319], [114 / 255] * 3)
```

- [ ] **Step 2: Run the new preprocessing tests and observe RED**

Run:

```powershell
python -m unittest tests.test_ai_detection.DetectionFunctionTests -v
```

Expected: import failure for the new transform functions and the native frame
is rejected by the old fixed-320 validator.

- [ ] **Step 3: Implement exact letterbox planning and NCHW preprocessing**

Add the immutable boundary and exact rounding policy:

```python
@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    input_size: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


def build_letterbox_transform(
    source_width: int,
    source_height: int,
    input_size: int = 320,
) -> LetterboxTransform:
    input_size = _validated_input_size(input_size)
    if (type(source_width) is not int or type(source_height) is not int
            or source_width <= 0 or source_height <= 0):
        raise ValueError("AI frame dimensions must be positive integers")
    gain = min(input_size / source_width, input_size / source_height)
    resized_width = min(
        input_size, max(1, math.floor(source_width * gain + 0.5))
    )
    resized_height = min(
        input_size, max(1, math.floor(source_height * gain + 0.5))
    )
    horizontal = input_size - resized_width
    vertical = input_size - resized_height
    left = horizontal // 2
    top = vertical // 2
    return LetterboxTransform(
        source_width, source_height, input_size,
        resized_width, resized_height,
        left, top, horizontal - left, vertical - top,
    )
```

Validate `frame.ndim == 3`, `frame.shape[2] == 3`, positive axes, and
`frame.dtype == np.uint8`. Resize only the content rectangle with Task 1,
copy it into `np.full((N, N, 3), 114, dtype=np.uint8)`, then keep the current
contiguous normalized float32 NCHW conversion.

- [ ] **Step 4: Add failing shared inverse-mapping tests**

Add literal legacy fixtures and call the shared mapper for both decoder
contracts:

```python
def test_inverse_letterbox_maps_content_and_rejects_padding_only_boxes(self):
    transform = build_letterbox_transform(1920, 1080, 320)
    mapped = map_detection_to_source(
        Detection(32, 82, 96, 142, 0.9, 7), transform
    )
    self.assertEqual(
        (mapped.x1, mapped.y1, mapped.x2, mapped.y2),
        (192.0, 72.0, 576.0, 432.0),
    )
    self.assertIsNone(map_detection_to_source(
        Detection(20, 0, 100, 69, 0.9, 7), transform
    ))
    crossing = map_detection_to_source(
        Detection(0, 60, 320, 100, 0.9, 7), transform
    )
    self.assertEqual(
        (crossing.x1, crossing.y1, crossing.x2, crossing.y2),
        (0.0, 0.0, 1920.0, 180.0),
    )

def test_inverse_uses_integer_resized_height_for_odd_padding(self):
    transform = build_letterbox_transform(640, 361, 320)
    mapped = map_detection_to_source(
        Detection(40, 79, 280, 239, 0.9, 7), transform
    )
    np.testing.assert_allclose(
        (mapped.x1, mapped.y1, mapped.x2, mapped.y2),
        (80.0, 19.94475138121547, 560.0, 339.060773480663),
        rtol=0,
        atol=1e-12,
    )

def test_landscape_inverse_mapping_is_exact_at_every_model_size(self):
    cases = (
        (160, Detection(16, 41, 48, 71, 0.9, 7)),
        (320, Detection(32, 82, 96, 142, 0.9, 7)),
        (640, Detection(64, 164, 192, 284, 0.9, 7)),
    )
    for input_size, model_box in cases:
        with self.subTest(input_size=input_size):
            mapped = map_detection_to_source(
                model_box,
                build_letterbox_transform(1920, 1080, input_size),
            )
            self.assertEqual(
                (mapped.x1, mapped.y1, mapped.x2, mapped.y2),
                (192.0, 72.0, 576.0, 432.0),
            )

def test_portrait_inverse_mapping_is_exact_at_every_model_size(self):
    cases = (
        (160, Detection(41, 16, 71, 48, 0.9, 7)),
        (320, Detection(82, 32, 142, 96, 0.9, 7)),
        (640, Detection(164, 64, 284, 192, 0.9, 7)),
    )
    for input_size, model_box in cases:
        with self.subTest(input_size=input_size):
            mapped = map_detection_to_source(
                model_box,
                build_letterbox_transform(1080, 1920, input_size),
            )
            self.assertEqual(
                (mapped.x1, mapped.y1, mapped.x2, mapped.y2),
                (72.0, 192.0, 432.0, 576.0),
            )
```

Add an `OnnxDetector` test in which the same 1920-by-1080 source box is
encoded once as a legacy row and once as a raw `(cx, cy, width, height)`
candidate for each input size 160, 320, and 640. Assert equal mapped
coordinates, unchanged confidence, class 7 for legacy and class 0 for raw,
and one inference call per detector. Repeat padding-only rejection at each
size so neither decoder can bypass the shared content intersection.

- [ ] **Step 5: Run inverse-mapping tests and observe RED**

Run:

```powershell
python -m unittest tests.test_ai_detection tests.test_ai_yolo -v
```

Expected: current parsers scale to canonical 320 space and cannot apply the
new source transform, so the new native-coordinate assertions fail.

- [ ] **Step 6: Move both decoders to model space and map once**

In `parse_output`, replace `320 / input_size` scaling with clipping to
`0..input_size`. In `decode_single_class_yolo`, remove
`LOGICAL_FRAME_SIZE / input_size` and clip kept boxes to `0..input_size`.
Retain all numeric validation, stable NMS, confidence floor, maximum 300
survivors, and detector-order restoration.

Implement source mapping with content intersection before inverse scaling:

```python
def map_detection_to_source(
    detection: Detection,
    transform: LetterboxTransform,
) -> Detection | None:
    content_right = transform.pad_left + transform.resized_width
    content_bottom = transform.pad_top + transform.resized_height
    x1 = max(float(transform.pad_left), detection.x1)
    y1 = max(float(transform.pad_top), detection.y1)
    x2 = min(float(content_right), detection.x2)
    y2 = min(float(content_bottom), detection.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    x_scale = transform.source_width / transform.resized_width
    y_scale = transform.source_height / transform.resized_height
    return Detection(
        max(0.0, min(transform.source_width,
                     (x1 - transform.pad_left) * x_scale)),
        max(0.0, min(transform.source_height,
                     (y1 - transform.pad_top) * y_scale)),
        max(0.0, min(transform.source_width,
                     (x2 - transform.pad_left) * x_scale)),
        max(0.0, min(transform.source_height,
                     (y2 - transform.pad_top) * y_scale)),
        detection.confidence,
        detection.class_id,
    )
```

Have `OnnxDetector.detect` prepare the tensor and transform together, decode
to model space, map each detection with this helper, and omit `None` results.
Do not change contract validation or provider construction.

- [ ] **Step 7: Run Task 2 verification**

Run:

```powershell
python -m unittest tests.test_image_resize tests.test_ai_detection tests.test_ai_yolo -v
```

Expected: all pass for 160/320/640, landscape, portrait, odd padding,
padding-only rejection, cross-padding clipping, raw/legacy equivalence, and
existing contract/provider safety.

- [ ] **Step 8: Commit Task 2**

```powershell
git add -- jitter_app/ai/detection.py jitter_app/ai/yolo.py tests/test_ai_detection.py tests/test_ai_yolo.py
git commit -m "feat: map letterboxed detections to source frames"
```

---

### Task 3: Frame-Aware Target Selection and Motion Response

**Files:**
- Modify: `jitter_app/ai/targeting.py`
- Test: `tests/test_ai_targeting.py`
- Verify unchanged consumers: `tests/test_combined_motion.py`
- Verify unchanged consumers: `tests/test_makcu_service.py`

**Interfaces:**
- Appends `frame_width: int = 320` and `frame_height: int = 320` to both
  `TargetSnapshot` and `DetectionFrameSnapshot`.
- Extends `analyze_detections` and `select_target` with keyword-only
  `frame_width: int = 320` and `frame_height: int = 320`.
- Keeps `AimMovementEngine.step(snapshot, settings, now)` unchanged while
  deriving center and response radius from each snapshot.
- Leaves legacy `tracking.py` on the appended 320 defaults; it is compatibility
  code and not a production selection path.

- [ ] **Step 1: Add failing native-center selection tests**

```python
def test_full_hd_analysis_selects_nearest_to_actual_frame_center(self):
    result = analyze_detections(
        (
            Detection(930, 500, 990, 580, 0.9, 7),
            Detection(930, 920, 990, 980, 0.9, 7),
        ),
        AimSettings(),
        sequence=8,
        captured_at=10.0,
        frame_width=1920,
        frame_height=1080,
    )
    self.assertEqual((result.target.aim_x, result.target.aim_y), (960.0, 540.0))
    self.assertEqual((result.target.frame_width, result.target.frame_height),
                     (1920, 1080))
    self.assertEqual((result.frame.frame_width, result.frame.frame_height),
                     (1920, 1080))
    self.assertEqual(result.frame.selected_index, 0)

def test_native_exact_distance_tie_preserves_detector_order(self):
    result = analyze_detections(
        (
            Detection(900, 510, 940, 570, 0.9, 7),
            Detection(980, 510, 1020, 570, 0.9, 7),
        ),
        AimSettings(),
        sequence=9,
        captured_at=11.0,
        frame_width=1920,
        frame_height=1080,
    )
    self.assertEqual(result.frame.selected_index, 0)

def test_analysis_rejects_invalid_frame_dimensions(self):
    for width, height in ((0, 1080), (1920, 0), (True, 1080), (1920, 1.5)):
        with self.subTest(width=width, height=height):
            with self.assertRaisesRegex(ValueError, "positive integers"):
                analyze_detections((), AimSettings(), sequence=1,
                                   captured_at=1.0,
                                   frame_width=width, frame_height=height)
```

- [ ] **Step 2: Run selection tests and observe RED**

Run:

```powershell
python -m unittest tests.test_ai_targeting.TargetSelectionTests -v
```

Expected: `analyze_detections` rejects the new keywords or still compares to
fixed `(160,160)`.

- [ ] **Step 3: Append snapshot geometry and generalize selection**

Use trailing defaults to preserve every existing positional fixture:

```python
@dataclass(frozen=True)
class TargetSnapshot:
    sequence: int
    captured_at: float
    target_class: str
    aim_x: float
    aim_y: float
    frame_width: int = 320
    frame_height: int = 320


@dataclass(frozen=True)
class DetectionFrameSnapshot:
    sequence: int
    captured_at: float
    detections: tuple[Detection, ...]
    selected_index: int | None
    frame_width: int = 320
    frame_height: int = 320
```

Validate exact positive integer dimensions at analysis entry, compare every
candidate against `(frame_width / 2, frame_height / 2)`, and pass both values
to the target and frame snapshots. Forward both keywords through
`select_target`.

- [ ] **Step 4: Add failing geometry-aware servo tests**

```python
def test_geometry_change_resets_old_velocity_fraction_and_error(self):
    engine = AimMovementEngine(nominal_hz=1000.0)
    settings = AimSettings(smoothing=0.0, aim_strength=0.35, max_step=20)
    old = TargetSnapshot(1, 10.0, "head", 320.0, 320.0, 320, 320)
    for tick in range(20):
        engine.step(old, settings, 10.0 + tick / 1000.0)

    replacement = TargetSnapshot(
        2, 10.020, "head", 960.0, 440.0, 1920, 1080
    )
    actual = [
        engine.step(replacement, settings, 10.020 + tick / 1000.0)
        for tick in range(10)
    ]
    clean = AimMovementEngine(nominal_hz=1000.0)
    expected = [
        clean.step(replacement, settings, 10.020 + tick / 1000.0)
        for tick in range(10)
    ]
    self.assertEqual(actual, expected)

def test_invalid_target_geometry_resets_and_emits_zero(self):
    engine = AimMovementEngine(nominal_hz=1000.0)
    settings = AimSettings(smoothing=0.0)
    engine.step(TargetSnapshot(1, 10.0, "head", 200, 160), settings, 10.0)

    self.assertEqual(
        engine.step(TargetSnapshot(2, 10.001, "head", 10, 10, 0, 1080),
                    settings, 10.001),
        (0, 0),
    )
    self.assertLessEqual(
        engine.step(TargetSnapshot(3, 10.002, "head", 154, 160),
                    settings, 10.002)[0],
        0,
    )

def test_response_curve_uses_native_half_radius_and_corner(self):
    engine = AimMovementEngine(nominal_hz=60.0)
    settings = AimSettings(
        smoothing=0.0,
        aim_strength=0.05,
        max_step=127,
        response_curve=(0.0, 0.0, 0.0, 0.0, 1.0),
    )
    radius = math.hypot(960.0, 540.0)
    half_radius = TargetSnapshot(
        1, 10.0, "head", 960.0 + radius / 2.0, 540.0, 1920, 1080
    )
    self.assertEqual(engine.step(half_radius, settings, 10.0), (0, 0))

    engine.reset()
    corner = TargetSnapshot(
        2, 10.1, "head", 1920.0, 1080.0, 1920, 1080
    )
    dx, dy = engine.step(corner, settings, 10.1)
    self.assertGreater(dx, 0)
    self.assertGreater(dy, 0)

def test_same_normalized_distance_scales_output_by_each_frame_radius(self):
    settings = AimSettings(
        smoothing=0.0,
        aim_strength=0.01,
        max_step=127,
        response_curve=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    cases = (
        (320, 320, 10.0, 1),
        (1920, 1080, 20.0, 5),
    )
    for width, height, now, expected_dx in cases:
        with self.subTest(size=(width, height)):
            center_x = width / 2.0
            center_y = height / 2.0
            radius = math.hypot(center_x, center_y)
            target = TargetSnapshot(
                1, now, "head", center_x + radius / 2.0, center_y,
                width, height,
            )
            engine = AimMovementEngine(nominal_hz=60.0)
            self.assertEqual(engine.step(target, settings, now), (expected_dx, 0))
```

Retain every exact existing 320 movement assertion. The half-radius test must
fail under the old fixed 320 radius because that implementation clamps the
native half-radius target to the curve's 100% point.

- [ ] **Step 5: Run servo tests and observe RED**

Run:

```powershell
python -m unittest tests.test_ai_targeting.AimMovementEngineTests -v
```

Expected: the engine uses fixed center/radius and retains state across frame
geometry changes.

- [ ] **Step 6: Implement dynamic center/radius without weakening bounds**

Before the settled-sequence early return, validate snapshot dimensions. On a
new geometry, clear velocity, fractions, remaining error, last timing, and the
settled tombstone, then consume the current snapshot as fresh. For each fresh
target use:

```python
center_x = snapshot.frame_width / 2.0
center_y = snapshot.frame_height / 2.0
next_remaining_x = snapshot.aim_x - center_x
next_remaining_y = snapshot.aim_y - center_y
reference_radius = math.hypot(center_x, center_y)
normalized = min(1.0, radius / reference_radius)
curve_distance = (
    response_curve_value(settings.response_curve, normalized)
    * reference_radius
)
```

Store current geometry in resettable engine state. Keep the current direction-
safe fractional filtering after geometry handling, and leave acceleration,
time-based smoothing, per-axis clamp, dead zone, expiry, and excess discard
unchanged.

- [ ] **Step 7: Run Task 3 verification**

```powershell
python -m unittest tests.test_ai_targeting tests.test_combined_motion tests.test_motion tests.test_makcu_service -v
```

Expected: native geometry tests and all exact 1,000 Hz/fractional-carry
regressions pass.

- [ ] **Step 8: Commit Task 3**

```powershell
git add -- jitter_app/ai/targeting.py tests/test_ai_targeting.py
git commit -m "feat: aim from native frame geometry"
```

---

### Task 4: Native-Aspect Adaptive Zoom

**Files:**
- Modify: `jitter_app/ai/zoom.py`
- Test: `tests/test_ai_zoom.py`
- Test: `tests/test_ai_detection.py`

**Interfaces:**
- Consumes frame dimensions appended in Task 3 and source-local detections from
  Task 2.
- Replaces square transform semantics with
  `ZoomTransform(left, top, crop_width, crop_height, source_width,
  source_height, factor)`.
- Keeps `build_zoom_input(frame, target, factor)` but returns an owned native-
  aspect crop instead of a resized 320 square.
- Keeps `select_zoom_factor`, `observe_zoom_stability`, `map_detection`,
  `map_target`, and `compose_zoom_refinement` names/signatures, with geometry
  derived from snapshots/transforms.

- [ ] **Step 1: Add failing widescreen crop and translation tests**

```python
def test_full_hd_two_x_crop_preserves_aspect_and_translates_back(self):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    target = TargetSnapshot(
        4, 20.0, "head", 960.0, 540.0, 1920, 1080
    )

    crop, transform = build_zoom_input(frame, target, 2.0)

    self.assertEqual(crop.shape, (540, 960, 3))
    self.assertTrue(crop.flags.c_contiguous)
    self.assertTrue(crop.flags.owndata)
    self.assertEqual(
        transform,
        ZoomTransform(480, 270, 960, 540, 1920, 1080, 2.0),
    )
    self.assertEqual(
        map_detection(Detection(10, 20, 110, 220, 0.9, 7), transform),
        Detection(490, 290, 590, 490, 0.9, 7),
    )

def test_native_crop_sizes_use_half_up_rounding_at_each_zoom_factor(self):
    cases = (
        ((1080, 1920), 1.5, (720, 1280)),
        ((1079, 1919), 2.0, (540, 960)),
    )
    for (height, width), factor, expected_shape in cases:
        with self.subTest(size=(width, height), factor=factor):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            target = TargetSnapshot(
                4, 20.0, "head", width / 2.0, height / 2.0,
                width, height,
            )
            crop, transform = build_zoom_input(frame, target, factor)
            self.assertEqual(crop.shape[:2], expected_shape)
            self.assertEqual(
                (transform.crop_width, transform.crop_height),
                (expected_shape[1], expected_shape[0]),
            )

def test_native_crop_rejects_target_from_different_frame_geometry(self):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    stale_target = TargetSnapshot(4, 20.0, "head", 160, 160)
    with self.assertRaisesRegex(ValueError, "match zoom source"):
        build_zoom_input(frame, stale_target, 2.0)

def test_native_zoom_crop_clamps_each_source_edge(self):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cases = (
        ((20.0, 20.0), (0, 0)),
        ((1900.0, 20.0), (960, 0)),
        ((20.0, 1060.0), (0, 540)),
        ((1900.0, 1060.0), (960, 540)),
    )
    for (aim_x, aim_y), (left, top) in cases:
        with self.subTest(aim=(aim_x, aim_y)):
            target = TargetSnapshot(
                5, 21.0, "head", aim_x, aim_y, 1920, 1080
            )
            _crop, transform = build_zoom_input(frame, target, 2.0)
            self.assertEqual((transform.left, transform.top), (left, top))
```

- [ ] **Step 2: Run geometry tests and observe RED**

```powershell
python -m unittest tests.test_ai_zoom.ZoomGeometryTests -v
```

Expected: fixed `(320,320,3)` validation rejects the native frame and the old
transform lacks rectangular/source fields.

- [ ] **Step 3: Implement native rectangular crop and direct translation**

Use positive half-up crop sizes and origin clamping:

```python
@dataclass(frozen=True)
class ZoomTransform:
    left: int
    top: int
    crop_width: int
    crop_height: int
    source_width: int
    source_height: int
    factor: float


crop_width = max(1, math.floor(source_width / factor + 0.5))
crop_height = max(1, math.floor(source_height / factor + 0.5))
left = max(0, min(source_width - crop_width,
                  math.floor(target.aim_x - crop_width / 2 + 0.5)))
top = max(0, min(source_height - crop_height,
                 math.floor(target.aim_y - crop_height / 2 + 0.5)))
crop = np.ascontiguousarray(
    frame[top:top + crop_height, left:left + crop_width].copy()
)
```

Require target dimensions to equal `frame.shape[1]` and `frame.shape[0]`, raising
`ValueError("Target dimensions must match zoom source")` when they differ.
Map crop-local detections and targets by adding `left`/`top`, clamp against
`source_width`/`source_height`, and discard mapped empty boxes. Do not resize
inside `zoom.py`; Task 2's detector owns letterboxing.

- [ ] **Step 4: Add failing logical-320 policy tests**

Add equivalent-resolution cases:

```python
def test_zoom_factor_uses_resolution_independent_logical_policy_space(self):
    legacy_target = TargetSnapshot(1, 10.0, "head", 160, 160)
    native_target = TargetSnapshot(2, 10.0, "head", 960, 540, 1920, 1080)
    legacy_box = Detection(150, 150, 170, 168, 0.9, 7)
    native_box = Detection(900, 480, 1020, 588, 0.9, 7)
    self.assertEqual(select_zoom_factor(legacy_box, legacy_target), 2.0)
    self.assertEqual(select_zoom_factor(native_box, native_target), 2.0)

def test_zoom_stability_scales_native_displacement_to_logical_320(self):
    first = TargetSnapshot(1, 10.0, "head", 960, 540, 1920, 1080)
    exact = TargetSnapshot(2, 10.01, "head", 1068, 540, 1920, 1080)
    outside = TargetSnapshot(3, 10.02, "head", 1068.1, 540, 1920, 1080)
    state = observe_zoom_stability(ZoomStabilityState(), first, 10.0)
    self.assertEqual(observe_zoom_stability(state, exact, 10.01).stable_count, 2)
    self.assertEqual(observe_zoom_stability(state, outside, 10.02).stable_count, 1)

def test_zoom_stability_resets_when_frame_geometry_changes(self):
    first = TargetSnapshot(1, 10.0, "head", 160, 160, 320, 320)
    changed = TargetSnapshot(2, 10.01, "head", 960, 540, 1920, 1080)
    state = observe_zoom_stability(ZoomStabilityState(), first, 10.0)

    result = observe_zoom_stability(state, changed, 10.01)

    self.assertEqual(result.previous_base_target, changed)
    self.assertEqual(result.stable_count, 1)
    self.assertAlmostEqual(result.cooldown_until, 10.11)
```

For composition, construct a full-HD base snapshot and a crop-local refined
box. Assert the selected replacement and refined target retain
`frame_width=1920`, `frame_height=1080`, unrelated base boxes remain byte-for-
byte equal. Parameterize landscape `(1920,1080)` and portrait `(1080,1920)`
cases and prove the fixed association boundary in source coordinates: an aim
point exactly `12.0 / policy_scale` beyond the seed edge is accepted and one
`0.01` source pixel farther is rejected. For both geometries
`policy_scale = 320 / 1920`, so the exact accepted source margin is `72.0`.

- [ ] **Step 5: Run policy/composition tests and observe RED**

```powershell
python -m unittest tests.test_ai_zoom tests.test_ai_detection -v
```

Expected: fixed 320 center, height, stability, square mapping, and association
margins fail the native-equivalence assertions.

- [ ] **Step 6: Generalize zoom policy while retaining exact boundaries**

Use this uniform policy scale:

```python
policy_scale = FRAME_SIZE / max(frame_width, frame_height)
policy_dx = (target.aim_x - frame_width / 2.0) * policy_scale
policy_dy = (target.aim_y - frame_height / 2.0) * policy_scale
policy_height = (detection.y2 - detection.y1) * policy_scale
```

Apply it to the existing center-distance, box-height, stability-displacement,
and fixed association-margin boundaries. Compute the latter explicitly as
`fixed_source_margin = 12.0 / policy_scale` before combining it with the
20-percent source-box margin. Preserve class compatibility,
confidence filtering, 1.5x/2.0x confirmation, cooldown, miss behavior, same-
frame fallback, and detector-order tie behavior. Publish base frame dimensions
on every composed target/frame snapshot. Treat different previous/current
frame dimensions as immediately unstable before measuring displacement.

- [ ] **Step 7: Run Task 4 verification**

```powershell
python -m unittest tests.test_ai_zoom tests.test_ai_detection tests.test_ai_targeting -v
```

Expected: all native and exact 320 zoom tests pass.

- [ ] **Step 8: Commit Task 4**

```powershell
git add -- jitter_app/ai/zoom.py tests/test_ai_zoom.py tests/test_ai_detection.py
git commit -m "feat: refine targets in native frame geometry"
```

---

### Task 5: AI Service Geometry Publication

**Files:**
- Modify: `jitter_app/ai/service.py`
- Test: `tests/test_ai_service.py`

**Interfaces:**
- Consumes arbitrary valid native frames and source-local detections from
  Tasks 2-4.
- Passes `frame_width` and `frame_height` into every base analysis.
- Preserves both dimensions through atomic publication, `reset_targeting`,
  targeting-revision invalidation, refinement, and generation barriers.
- Keeps one base call and at most one gated refinement call.

- [ ] **Step 1: Make service fakes represent real RGB frames**

Before production changes, add one complete RGB fixture helper and update the
shared fakes so lifecycle tests do not accidentally use scalar/object frames
while retaining their existing coordinate signal:

```python
def rgb_frame(value=0, *, width=320, height=320):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[0, 0] = value
    return frame


class FakeCapture:
    def __init__(self, frames=None):
        self.frames = list(
            frames if frames is not None else [rgb_frame(10)]
        )
        self.closed = threading.Event()


class FakeDetector:
    provider = "DmlExecutionProvider"

    def __init__(self):
        self.calls = []

    def detect(self, frame):
        self.calls.append(frame)
        value = float(frame[0, 0, 0])
        return (Detection(value, value, value + 10, value + 10, 0.9, 7),)
```

Keep the existing `start`, `read`, and `close` methods below the shown
`FakeCapture.__init__`. Replace explicit successful `10`, `20`, and `object()`
frame payloads with `rgb_frame(<original scalar>)`. Use `rgb_frame()` where an
object carried no coordinate meaning. Keep `None` only where a test
intentionally means "no latest frame." Preserve detector/capture
synchronization events and counters.

- [ ] **Step 2: Run service tests after fixture normalization**

```powershell
python -m unittest tests.test_ai_service -v
```

Expected: all current tests still pass before service production behavior is
changed. If a test used scalar identity as its intended signal, encode that
signal in a pixel value and have its dedicated fake inspect that literal.

- [ ] **Step 3: Add failing native publication tests**

```python
def test_native_frame_dimensions_publish_atomically_with_target_and_boxes(self):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    detector = SequenceDetector([
        (Detection(300, 160, 340, 220, 0.9, 7),),
    ])
    service = AiService(
        detector_factory=lambda _path: detector,
        capture_factory=lambda: FakeCapture([frame]),
        event_sink=lambda _event: None,
    )
    self.addCleanup(service.close)

    service.start(lambda: AimSettings(), lambda: False)
    self.assertTrue(wait_until(lambda: service.latest_snapshot() is not None))

    target = service.latest_snapshot()
    boxes = service.latest_detection_snapshot()
    self.assertEqual((target.frame_width, target.frame_height), (640, 360))
    self.assertEqual((boxes.frame_width, boxes.frame_height), (640, 360))
    self.assertEqual((target.aim_x, target.aim_y), (320.0, 190.0))
    self.assertEqual(detector.calls, 1)
```

Extend the current `reset_targeting` and targeting-revision race tests to
assert that clearing `selected_index` preserves the exact frame dimensions.

Add a `ControlledCapture` two-frame test with `(360,640,3)` followed by
`(600,800,3)`, normal zoom gate false, and a `SequentialDetector` that returns
one target per base call. Release only the first permit, wait for sequence 1,
and assert both snapshots carry `(640,360)`. Release only the second permit,
wait for sequence 2, and assert both snapshots carry `(800,600)`. Assert the
detector frame shapes are exactly `[(360,640,3), (600,800,3)]`; this prevents
dimensions from being cached or paired with the wrong sequence.

Add an exactly eligible native zoom test with two controlled `(360,640,3)`
frames, a mutable clock at `10.0` then `10.11`, and these detector results in
call order:

```python
base = (Detection(302, 162, 338, 198, 0.90, 7),)
refined_one_half = (Detection(195, 102, 231, 138, 0.93, 7),)
refined_two = (Detection(142, 72, 178, 108, 0.94, 7),)
detector = SequentialDetector((
    base, refined_one_half,
    base, refined_two,
))
```

The first observation is confirmation-capped to 1.5x and the second is
confirmed after cooldown for 2.0x. Assert exact detector shapes, in order:
`[(360,640,3), (240,427,3), (360,640,3), (180,320,3)]`, and exact zoom events
`[1.5, 2.0]`. Keep the existing Overlay-only, idle, and Test 3s cases asserting
exactly one base call and zero refinement calls.

- [ ] **Step 4: Run new service tests and observe RED**

```powershell
python -m unittest tests.test_ai_service -v
```

Expected: published snapshots still default to 320x320 and rebuilt frames lose
native geometry.

- [ ] **Step 5: Thread geometry through every service publication path**

Immediately after a valid capture read, derive:

```python
frame_height, frame_width = frame.shape[:2]
base_analysis = analyze_detections(
    base_detections,
    settings,
    sequence=sequence,
    captured_at=captured_at,
    frame_width=frame_width,
    frame_height=frame_height,
)
```

When rebuilding `DetectionFrameSnapshot` in `reset_targeting` or after a
targeting revision change, append `frame.frame_width` and
`frame.frame_height`. Keep atomic lock publication and all current-generation
checks before/after base and refinement calls. Do not add an inference call or
wait for movement cadence.

- [ ] **Step 6: Run Task 5 verification**

```powershell
python -m unittest tests.test_ai_service tests.test_ai_zoom tests.test_ai_targeting -v
```

Expected: native base/refinement geometry, call counts, fallback, resets,
generation invalidation, FPS accounting, and 320 compatibility all pass.

- [ ] **Step 7: Commit Task 5**

```powershell
git add -- jitter_app/ai/service.py tests/test_ai_service.py
git commit -m "feat: publish ai frame geometry atomically"
```

---

### Task 6: Activate Full-Output Capture and Full-Screen Overlay

**Files:**
- Modify: `AGENTS.md`
- Modify: `jitter_app/ai/capture.py`
- Modify: `jitter_app/presentation/overlay.py`
- Test: `tests/test_ai_capture.py`
- Test: `tests/test_ai_service.py`
- Test: `tests/test_overlay.py`

**Interfaces:**
- Replaces `centered_region` with
  `full_output_region(width, height) -> tuple[int, int, int, int]`.
- Changes `DxcamCapture.read()` to accept any non-empty native RGB `uint8`
  shape and return an owned contiguous array without dtype coercion.
- Extends `project_overlay_boxes` with optional keyword-only
  `canvas_width`/`canvas_height`; when omitted it projects 1:1 in snapshot
  coordinates for pure compatibility tests.
- Removes `CAPTURE_SIZE`, `_capture_left`, and `_capture_top` from Overlay
  rendering. Production render always passes current screen dimensions.

- [ ] **Step 1: Add failing full-output capture tests**

Replace centered-region expectations with:

```python
from jitter_app.ai.capture import DxcamCapture, full_output_region

def test_full_output_region_requires_positive_integer_geometry(self):
    self.assertEqual(full_output_region(1920, 1080), (0, 0, 1920, 1080))
    for width, height in ((0, 1080), (1920, 0), (True, 1080),
                          (1920, 1.5)):
        with self.subTest(width=width, height=height):
            with self.assertRaisesRegex(ValueError, "positive integers"):
                full_output_region(width, height)

def test_capture_requests_full_output_and_returns_owned_native_rgb(self):
    source = np.zeros((1080, 1920, 3), dtype=np.uint8)
    camera = FakeCamera(width=1920, height=1080, frame=source)
    factory = RecordingCameraFactory(camera)
    capture = DxcamCapture(camera_factory=factory, target_fps=165)

    capture.start()
    frame = capture.read()

    self.assertEqual(camera.start_kwargs, {
        "region": (0, 0, 1920, 1080), "target_fps": 165,
    })
    self.assertEqual(frame.shape, (1080, 1920, 3))
    self.assertEqual(frame.dtype, np.uint8)
    self.assertTrue(frame.flags.c_contiguous)
    self.assertTrue(frame.flags.owndata)
    self.assertFalse(np.shares_memory(frame, source))
```

Assert `None` still means “no new frame.” Extend malformed frames with empty
axes and float/bool RGB arrays; every non-`None` malformed case must raise
`ValueError("AI capture frame must be nonempty RGB uint8")`, not be coerced or
silently retried.

- [ ] **Step 2: Run capture tests and observe RED**

```powershell
python -m unittest tests.test_ai_capture -v
```

Expected: missing `full_output_region`, centered start arguments, and fixed
320 shape validation fail.

- [ ] **Step 3: Implement strict full-output capture**

```python
def full_output_region(width: int, height: int) -> tuple[int, int, int, int]:
    if (type(width) is not int or type(height) is not int
            or width <= 0 or height <= 0):
        raise ValueError("Primary output dimensions must be positive integers")
    return 0, 0, width, height
```

Start DXCam with that region and unchanged target FPS. In `read`, return `None`
only when DXCam returns `None`. For any non-`None` value, require an
`np.ndarray` with `ndim == 3`, positive height/width, exactly three channels,
and `dtype == np.uint8`; otherwise raise the exact `ValueError` above so the
service enters its existing AI runtime-error path. Return valid pixels through
`np.ascontiguousarray(frame.copy())`. Preserve idempotent stop/release cleanup.

- [ ] **Step 4: Add failing source-to-canvas Overlay tests**

Replace the centered-region rendering assertions with:

```python
def test_render_projects_full_source_frame_over_entire_canvas(self):
    overlay, _window, canvases, _adapter, _calls = self.make_overlay(
        screen_size=(2560, 1440)
    )
    overlay.show()
    frame = DetectionFrameSnapshot(
        1, 10.0,
        (Detection(0, 0, 2560, 1440, 0.8, 0),),
        0,
        2560,
        1440,
    )
    overlay.render(frame, now=10.0)
    self.assertEqual(canvases[0].items[0][0], (0, 0, 2560, 1440))

def test_projection_scales_snapshot_geometry_to_changed_canvas(self):
    frame = DetectionFrameSnapshot(
        1, 10.0,
        (Detection(480, 270, 1440, 810, 0.8, 0),),
        0,
        1920,
        1080,
    )
    boxes = project_overlay_boxes(
        frame,
        10.0,
        canvas_width=2560,
        canvas_height=1440,
    )
    self.assertEqual(
        (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2),
        (640.0, 360.0, 1920.0, 1080.0),
    )

def test_render_passes_changed_canvas_geometry_into_projection(self):
    overlay, _window, canvases, _adapter, _calls = self.make_overlay(
        screen_size=(2560, 1440)
    )
    overlay.show()
    frame = DetectionFrameSnapshot(
        1, 10.0,
        (Detection(480, 270, 1440, 810, 0.8, 0),),
        0,
        1920,
        1080,
    )

    overlay.render(frame, now=10.0)

    self.assertEqual(
        canvases[0].items[0][0],
        (640.0, 360.0, 1920.0, 1080.0),
    )

def test_projection_clamps_boxes_and_labels_to_canvas(self):
    frame = DetectionFrameSnapshot(
        1, 10.0,
        (Detection(-20, -10, 1940, 1090, 0.8, 7),),
        0,
        1920,
        1080,
    )
    boxes = project_overlay_boxes(
        frame, 10.0, canvas_width=1920, canvas_height=1080,
        label_mode="class",
    )
    self.assertEqual(
        (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2,
         boxes[0].label),
        (0.0, 0.0, 1920.0, 1080.0, "HEAD"),
    )

def test_invalid_canvas_and_empty_projection_preserve_original_selection(self):
    frame = DetectionFrameSnapshot(
        1, 10.0,
        (
            Detection(-20, 10, -1, 40, 0.8, 0),
            Detection(100, 100, 200, 200, 0.9, 7),
        ),
        1,
        1920,
        1080,
    )
    for width, height in ((0, 1080), (-1, 1080), (True, 1080),
                          (1920, 0), (1920, False)):
        with self.subTest(canvas=(width, height)):
            self.assertEqual(project_overlay_boxes(
                frame, 10.0, canvas_width=width, canvas_height=height,
            ), ())

    boxes = project_overlay_boxes(
        frame, 10.0, canvas_width=1920, canvas_height=1080,
    )
    self.assertEqual(len(boxes), 1)
    self.assertEqual(boxes[0].width, 4)

    selected_was_omitted = DetectionFrameSnapshot(
        1, 10.0, frame.detections, 0, 1920, 1080,
    )
    boxes = project_overlay_boxes(
        selected_was_omitted, 10.0,
        canvas_width=1920, canvas_height=1080,
    )
    self.assertEqual(len(boxes), 1)
    self.assertEqual(boxes[0].width, 2)
```

Update render fixtures with actual source dimensions so expected box locations
are direct screen positions rather than centered offsets. Retain every style,
label, selected-width, HUD, stale lock, transparency-key, setup, cleanup, and
main-thread assertion. Add a four-case label-bound test using a 100x100 fake
canvas whose `bbox_value` is respectively `(-10,10,30,30)`,
`(80,10,110,30)`, `(10,-5,30,15)`, and `(10,85,30,105)`. Render one labeled
box with no runtime HUD and assert the label item is moved by `(10,0)`,
`(-10,0)`, `(0,5)`, and `(0,-5)` respectively.

In `tests/test_ai_service.py`, add a `CountingCapture` whose `read_error` is
`ValueError("AI capture frame must be nonempty RGB uint8")`. Start the service,
wait for its `error` event, then assert the capture is closed exactly once,
both latest snapshots are cleared, and the event contains a concise AI runtime
failure without a traceback. Retain the existing UI error assertions that hide
Overlay, deselect AI Aim, and preserve eligible Jitter movement.

- [ ] **Step 5: Run Overlay tests and observe RED**

```powershell
python -m unittest tests.test_overlay -v
```

Expected: project lacks canvas geometry and render still adds centered
320 offsets.

- [ ] **Step 6: Implement pure projection and remove centered offsets**

Inside `project_overlay_boxes`, default omitted canvas dimensions to the
snapshot dimensions, reject non-positive/invalid geometry with an empty tuple,
and project with:

```python
x_scale = canvas_width / snapshot.frame_width
y_scale = canvas_height / snapshot.frame_height
x1 = max(0.0, min(float(canvas_width), detection.x1 * x_scale))
y1 = max(0.0, min(float(canvas_height), detection.y1 * y_scale))
x2 = max(0.0, min(float(canvas_width), detection.x2 * x_scale))
y2 = max(0.0, min(float(canvas_height), detection.y2 * y_scale))
```

Omit empty projected boxes while preserving original detection indices for
selected emphasis. In `DetectionOverlay.render`, pass `_screen_width` and
`_screen_height`, draw projected coordinates directly, and remove all capture
offset state. After creating each label item, call `_keep_item_on_screen` for
that item, just as the final HUD item is corrected; coordinate-only anchor
clamping is insufficient because the rendered text extent can cross an edge.

In this same activation commit, replace `AGENTS.md` statements that define the
production capture as centered/canonical 320 with the approved behavior:
capture the complete native primary output, letterbox each base/crop frame to
the active 160/320/640 model square, publish source-screen geometry, target the
real frame center, and render source coordinates across the full Overlay.
Retain the canonical 320 movement-policy wording only where it describes
resolution-independent threshold normalization. This keeps authoritative
repository guidance consistent with the code at every commit.

- [ ] **Step 7: Run Task 6 verification**

```powershell
python -m unittest tests.test_ai_capture tests.test_overlay tests.test_ai_service tests.test_ui -v
```

Expected: full-output capture, complete-canvas boxes, display-size refresh,
styles/HUD, service demand, STOP, and runtime-error behavior all pass.

- [ ] **Step 8: Commit Task 6**

```powershell
git add -- AGENTS.md jitter_app/ai/capture.py jitter_app/presentation/overlay.py tests/test_ai_capture.py tests/test_ai_service.py tests/test_overlay.py
git commit -m "feat: detect and render the full primary display"
```

---

### Task 7: Documentation, Regression Gate, and Runtime Review

**Files:**
- Modify: `README.md`
- Verify: all Python sources and tests

**Interfaces:**
- Consumes the completed Tasks 1-6.
- Produces documentation that describes native full-primary capture,
  model-input letterboxing, source-screen detections, dynamic response radius,
  and native-aspect Adaptive Zoom without changing config or packaging.

- [ ] **Step 1: Update repository documentation precisely**

Replace fixed-centered-capture statements with these semantics:

```text
DXCam captures the complete primary display at native resolution. Each base
frame is aspect-preserving letterboxed into the active 160, 320, or 640 square
model input, and detections are mapped back to source-screen coordinates.
The base path uses one inference per processed frame; eligible Adaptive Zoom
may add one same-frame refinement call. The 1,000 Hz motion servo remains
independent from capture and inference cadence.
```

Confirm the authoritative behavior statements in `AGENTS.md` were updated in
Task 6. In `README.md`, state explicitly that the bundled 320 model can lose
small-target detail over a full widescreen field, compatible external 640
models remain runtime-only, and no capture geometry is persisted. Update the
`capture.py`, `resize.py`, `detection.py`, `targeting.py`, `zoom.py`, and
`overlay.py` layout descriptions.
Do not add a UI control, schema change, dependency, model file, or packaging
argument.

- [ ] **Step 2: Run focused cross-component regression tests**

```powershell
python -m unittest tests.test_ai_capture tests.test_image_resize tests.test_ai_detection tests.test_ai_yolo tests.test_ai_targeting tests.test_ai_zoom tests.test_ai_service tests.test_overlay tests.test_ui tests.test_package_layout tests.test_distribution_metadata -v
```

Expected: all focused tests pass.

- [ ] **Step 3: Run canonical source compilation**

```powershell
$jitterSources = @('main.py', 'distribution_metadata.py') + @(Get-ChildItem -LiteralPath 'jitter_app' -Recurse -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
python -m py_compile @jitterSources
```

Expected: exit code 0 and no compile error.

- [ ] **Step 4: Run the complete unit suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 5: Verify pinned runtime imports**

```powershell
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
```

Expected: exit code 0 with no import error.

- [ ] **Step 6: Verify the bundled DirectML runtime**

```powershell
python .\main.py --ai-runtime-self-check
```

Expected: JSON status `ok`, bundled input size 320, the approved bundled model
hash, and `DmlExecutionProvider`.

- [ ] **Step 7: Review the canonical distribution plan without building**

```powershell
python .\distribution_metadata.py --review-json
```

Expected: exit code 0; only `models/all_games_320.onnx` is planned as model
data, and no external `.onnx` path appears.

- [ ] **Step 8: Check diff hygiene and user-owned files**

```powershell
git diff --check
git status --short
```

Expected: no whitespace error. Compare every pre-work untracked external ONNX
path and SHA-256 recorded in the SDD ledger against the original workspace;
the same path set remains untracked and every digest is unchanged, without
assuming a fixed model count. No `build-output/`, `dist/`, `*.build/`,
`*.dist/`, `__pycache__/`, or `app.log` source change is staged.

- [ ] **Step 9: Commit Task 7**

```powershell
git add -- README.md
git commit -m "docs: describe full-primary-display detection"
```

- [ ] **Step 10: Request independent whole-branch review**

Generate a review package from the feature-branch merge base through `HEAD`.
The reviewer must check spec compliance and code quality, with particular
attention to integer letterbox scales, raw/legacy equivalence, geometry resets,
double-transform risk in zoom, one-base/one-refinement call bounds, overlay
clamping, cancellation, config non-persistence, and preservation of the fixed
1,000 Hz servo. Resolve every Critical or Important finding through the
subagent-driven review loop before finishing the branch.
