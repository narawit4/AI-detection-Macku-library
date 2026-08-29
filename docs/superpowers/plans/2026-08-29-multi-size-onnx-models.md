# Multi-Size External ONNX Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow runtime-browsed external ONNX models with exact 160, 320, or 640 square inputs while preserving Jitter's canonical 320-by-320 capture, targeting, Overlay, movement, and Adaptive Zoom geometry.

**Architecture:** Extract the existing pure NumPy RGB resize into a shared module used by zoom and detection. `OnnxDetector` validates one of three static input sizes, resizes canonical frames to that input, and maps output boxes back to canonical 320-space; model validation carries the detected size to the UI without making runtime inference trust UI metadata.

**Tech Stack:** Python 3.11+, NumPy, ONNX Runtime DirectML, DXCam, Tkinter, `unittest`

**Spec:** `docs/superpowers/specs/2026-08-29-multi-size-onnx-models-design.md`

## Global Constraints

- The bundled startup-default remains `models/all_games_320.onnx` with its exact approved SHA-256 and DirectML self-check.
- External input shapes accepted are exactly `[1,3,160,160]`, `[1,3,320,320]`, and `[1,3,640,640]`; reject 128, 256, dynamic, rectangular, and arbitrary shapes.
- Input stays `images` `tensor(float)` and output stays `output0` `tensor(float)` `[1,300,6]`, with player class 0 and head class 7.
- Capture, Overlay, crosshair, target snapshots, response curve, movement, and Adaptive Zoom remain in canonical 320-by-320 coordinates.
- Every launch starts with the bundled 320 model; never persist, copy, download, bundle, or package an external model or its path/size.
- Do not add OpenCV, Pillow, Torch, Ultralytics, or another image/model runtime.
- Preserve off-UI-thread validation, exact-ready switching, rollback-once, generation safety, immediate cancellation, and AI failure isolation.
- Preserve unrelated user files, especially untracked external ONNX files in `models/`.
- Use test-driven development for every production change: observe RED, implement minimally, observe GREEN, then commit.
- Do not run Nuitka; packaging is not requested.

## File Structure

- Create `image_resize.py`: pure deterministic NumPy RGB bilinear resize and cached coordinate plans.
- Create `tests/test_image_resize.py`: isolated resize validation, regression hashes, output ownership, and 160/320/640 shape tests.
- Modify `ai_zoom.py`: import the shared resize helper and remove its duplicate resize implementation; keep all zoom behavior canonical.
- Modify `tests/test_ai_zoom.py`: remove helper-internal tests moved to `test_image_resize.py` and retain zoom geometry/composition coverage.
- Modify `tests/test_distribution_metadata.py`: include the new source module in the canonical compile target set as soon as it exists.
- Modify `tests/test_entrypoints.py`: include the new source module in review-json expectations, then later add README contract assertions.
- Modify `ai_detection.py`: validate supported static input sizes, resize canonical frames, expose `input_size`, and map output boxes to canonical coordinates.
- Modify `tests/test_ai_detection.py`: contract matrix, preprocessing sizes, output scaling/clipping, provider behavior, and canonical downstream behavior.
- Modify `ai_model_selection.py`: attach optional immutable input-size metadata to `ModelChoice` and publish a validated choice from `ModelValidator`.
- Modify `tests/test_ai_model_selection.py`: unknown-before-validation, known-after-validation, immutable metadata, stale event, error, and cancellation coverage.
- Modify `ui.py`: display validated model size and commit the validator's enriched choice through the existing generation-safe switch lifecycle.
- Modify `tests/test_ui.py`: labels, loading state, exact event matching, switch/rollback/cancellation, and non-persistence regressions.
- Modify `AGENTS.md`: document the shared resize module and accepted external sizes while retaining fixed canonical geometry.
- Modify `README.md`: document auto-detected 160/320/640 external inputs, fixed 320 source area, labels, rejections, and performance trade-offs.

---

### Task 1: Extract the Shared NumPy RGB Resize Primitive

**Files:**
- Create: `image_resize.py`
- Create: `tests/test_image_resize.py`
- Modify: `ai_zoom.py:1-220`
- Modify: `tests/test_ai_zoom.py:1-245`
- Modify: `tests/test_distribution_metadata.py:260-280`
- Modify: `tests/test_entrypoints.py:315-345`

**Interfaces:**
- Consumes: NumPy `uint8` arrays shaped `H x W x 3` and a positive integer `output_size`.
- Produces: `resize_rgb_bilinear(image: np.ndarray, output_size: int = 320) -> np.ndarray` and private cached `_resize_plan(source_height: int, source_width: int, output_size: int) -> tuple[np.ndarray, ...]` in `image_resize.py`.
- Preserves: `ai_zoom.resize_rgb_bilinear` remains importable through `from image_resize import resize_rgb_bilinear` at module scope, so existing callers do not break during extraction.

- [ ] **Step 1: Add isolated tests for the shared module and move resize-only assertions out of zoom tests**

Create `tests/test_image_resize.py` with the existing frozen pixel hashes and the requested model sizes:

```python
import hashlib
import unittest

import numpy as np

import image_resize
from image_resize import resize_rgb_bilinear


class RgbResizeTests(unittest.TestCase):
    def test_reuses_immutable_cached_coordinate_plan(self):
        image_resize._resize_plan.cache_clear()
        first = image_resize._resize_plan(160, 160, 320)
        second = image_resize._resize_plan(160, 160, 320)
        self.assertIs(first, second)
        for values in first:
            with self.assertRaises(ValueError):
                values.flags.writeable = True

    def test_zoom_size_regression_hashes_are_unchanged(self):
        expected_hashes = {
            160: "73fff29a5890f0cdad009470c7ade901489fcc096309af53489d1812cf23e5a5",
            213: "d3876b8a91d71fcd2303a1f5e94c551761191e534b8861c623a908a1c3a4bd32",
        }
        random = np.random.default_rng(20260827)
        for size, expected_hash in expected_hashes.items():
            with self.subTest(size=size):
                source = random.integers(0, 256, (size, size, 3), dtype=np.uint8)
                resized = resize_rgb_bilinear(source)
                self.assertEqual(
                    hashlib.sha256(resized.tobytes()).hexdigest(),
                    expected_hash,
                )

    def test_produces_owned_contiguous_uint8_outputs_for_supported_sizes(self):
        source = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        for output_size in (160, 320, 640):
            with self.subTest(output_size=output_size):
                resized = resize_rgb_bilinear(source, output_size)
                self.assertEqual(resized.shape, (output_size, output_size, 3))
                self.assertEqual(resized.dtype, np.uint8)
                self.assertTrue(resized.flags.c_contiguous)
                self.assertFalse(np.shares_memory(resized, source))

    def test_preserves_hand_derived_center_and_half_up_rounding(self):
        source = np.array(
            [[[0, 0, 0], [100, 100, 100]],
             [[200, 200, 200], [255, 255, 255]]],
            dtype=np.uint8,
        )
        resized = resize_rgb_bilinear(source, 3)
        self.assertEqual(resized[1, 1].tolist(), [139, 139, 139])

        boundary = np.array(
            [[[84, 84, 84], [244, 244, 244]],
             [[7, 7, 7], [191, 191, 191]]],
            dtype=np.uint8,
        )
        self.assertEqual(
            resize_rgb_bilinear(boundary, 7)[1, 1].tolist(),
            [99, 99, 99],
        )

    def test_rejects_malformed_source_and_output_size(self):
        bad_sources = (
            None,
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2, 4), dtype=np.uint8),
            np.zeros((0, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.float32),
        )
        for source in bad_sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "nonempty RGB uint8"):
                    resize_rgb_bilinear(source, 160)
        for output_size in (True, 0, -1, 1.5):
            with self.subTest(output_size=output_size):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    resize_rgb_bilinear(
                        np.zeros((2, 2, 3), dtype=np.uint8), output_size
                    )


if __name__ == "__main__":
    unittest.main()
```

Remove only the equivalent helper-internal tests from `ZoomGeometryTests`; keep crop, mapping, factor, stability, and composition tests in `tests/test_ai_zoom.py`.

In the expected compile-target sets in `tests/test_distribution_metadata.py`
and `tests/test_entrypoints.py`, add exactly:

```python
"image_resize.py",
```

- [ ] **Step 2: Run the new resize test to verify RED**

Run:

```powershell
python -m unittest tests.test_image_resize tests.test_distribution_metadata tests.test_entrypoints -v
```

Expected: import failure because `image_resize.py` does not exist and compile
target expectations name a source file that discovery cannot yet find.

- [ ] **Step 3: Move the existing implementation into `image_resize.py`**

Create `image_resize.py` with the exact pure interface and add optimized
separable cases for model-input resizing:

```python
"""Pure deterministic NumPy RGB resize shared by AI detection and zoom."""

from functools import lru_cache

import numpy as np


_SEPARABLE_RESIZE_SHAPES = {
    (160, 160, 320),
    (213, 213, 320),
    (320, 320, 160),
    (320, 320, 640),
}


@lru_cache(maxsize=16)
def _resize_plan(
    source_height: int,
    source_width: int,
    output_size: int,
) -> tuple[np.ndarray, ...]:
    source_x = np.linspace(0.0, source_width - 1, output_size)
    source_y = np.linspace(0.0, source_height - 1, output_size)
    x0 = np.floor(source_x).astype(np.intp)
    y0 = np.floor(source_y).astype(np.intp)
    values = (
        x0,
        np.minimum(x0 + 1, source_width - 1),
        y0,
        np.minimum(y0 + 1, source_height - 1),
        (source_x - x0)[None, :, None],
        (source_y - y0)[:, None, None],
    )
    return tuple(
        np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
        for value in values
    )


def resize_rgb_bilinear(
    image: np.ndarray,
    output_size: int = 320,
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
    x0, x1, y0, y1, wx, wy = _resize_plan(
        source_height, source_width, output_size
    )
    if (source_height, source_width, output_size) in _SEPARABLE_RESIZE_SHAPES:
        source = image.astype(np.float64)
        horizontal = source[:, x0, :] * (1.0 - wx) + source[:, x1, :] * wx
        blended = horizontal[y0, :, :] * (1.0 - wy) + horizontal[y1, :, :] * wy
    else:
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

In `ai_zoom.py`, delete `_resize_plan`, `_SEPARABLE_RESIZE_SHAPES`, and the local
`resize_rgb_bilinear`, remove the unused `lru_cache` import, and import:

```python
from image_resize import resize_rgb_bilinear
```

- [ ] **Step 4: Run shared resize and complete zoom tests to verify GREEN**

Run:

```powershell
python -m unittest tests.test_image_resize tests.test_ai_zoom tests.test_distribution_metadata tests.test_entrypoints -v
```

Expected: all resize regression hashes and all zoom behavior tests pass.

- [ ] **Step 5: Commit the shared resize extraction**

```powershell
git add image_resize.py ai_zoom.py tests/test_image_resize.py tests/test_ai_zoom.py tests/test_distribution_metadata.py tests/test_entrypoints.py
git commit -m "refactor: share deterministic RGB resize"
```

---

### Task 2: Accept 160/320/640 Detector Inputs and Map Output to 320-Space

**Files:**
- Modify: `ai_detection.py`
- Modify: `tests/test_ai_detection.py`

**Interfaces:**
- Consumes: `resize_rgb_bilinear(frame, input_size)` from Task 1.
- Produces: `SUPPORTED_INPUT_SIZES: tuple[int, ...] = (160, 320, 640)`, `preprocess_frame(frame: np.ndarray, input_size: int = 320) -> np.ndarray`, `parse_output(output: np.ndarray, input_size: int = 320) -> tuple[Detection, ...]`, and read-only `OnnxDetector.input_size: int`.
- Preserves: `OnnxDetector.detect(frame)` still consumes canonical `(320,320,3)` uint8 RGB and returns canonical `Detection` coordinates.

- [ ] **Step 1: Add failing detector contract, preprocessing, and coordinate mapping tests**

Extend `tests/test_ai_detection.py`:

```python
class DetectionFunctionTests(unittest.TestCase):
    def test_preprocess_resizes_canonical_frame_to_each_supported_input(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        frame[0, 0] = (255, 128, 0)
        for input_size in (160, 320, 640):
            with self.subTest(input_size=input_size):
                tensor = preprocess_frame(frame, input_size)
                self.assertEqual(tensor.shape, (1, 3, input_size, input_size))
                self.assertEqual(tensor.dtype, np.float32)
                self.assertTrue(tensor.flags.c_contiguous)
                np.testing.assert_allclose(
                    tensor[0, :, 0, 0], [1.0, 128 / 255, 0.0]
                )

    def test_output_coordinates_map_from_model_space_to_canonical_space(self):
        cases = (
            (160, (10, 20, 30, 40), (20.0, 40.0, 60.0, 80.0)),
            (320, (10, 20, 30, 40), (10.0, 20.0, 30.0, 40.0)),
            (640, (10, 20, 30, 40), (5.0, 10.0, 15.0, 20.0)),
        )
        for input_size, raw_box, expected in cases:
            with self.subTest(input_size=input_size):
                output = np.zeros((1, 300, 6), dtype=np.float32)
                output[0, 0] = (*raw_box, 0.75, 7)
                detection = parse_output(output, input_size)[0]
                self.assertEqual(
                    (detection.x1, detection.y1, detection.x2, detection.y2),
                    expected,
                )

    def test_output_scales_before_canonical_clipping_and_empty_rejection(self):
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, :2] = (
            (-20, -20, 80, 80, 0.8, 7),
            (-20, 2, -4, 20, 0.9, 0),
        )
        detections = parse_output(output, 640)
        self.assertEqual(len(detections), 1)
        self.assertEqual(
            (detections[0].x1, detections[0].y1,
             detections[0].x2, detections[0].y2),
            (0.0, 0.0, 40.0, 40.0),
        )


class OnnxDetectorTests(unittest.TestCase):
    def test_accepts_only_exact_supported_static_square_input_sizes(self):
        for input_size in (160, 320, 640):
            with self.subTest(input_size=input_size):
                session = Session(inputs=[NodeArg(
                    "images", "tensor(float)",
                    [1, 3, input_size, input_size],
                )])
                detector = OnnxDetector(
                    "model.onnx",
                    session_factory=lambda *_args, **_kwargs: session,
                )
                self.assertEqual(detector.input_size, input_size)

        rejected_shapes = (
            [1, 3, 128, 128],
            [1, 3, 256, 256],
            [1, 3, 160, 320],
            [1, 3, "height", "width"],
            [1, 160, 160, 3],
        )
        for shape in rejected_shapes:
            with self.subTest(shape=shape):
                session = Session(inputs=[NodeArg(
                    "images", "tensor(float)", shape
                )])
                with self.assertRaisesRegex(
                    ModelContractError, "160, 320, or 640"
                ):
                    OnnxDetector(
                        "model.onnx",
                        session_factory=lambda *_args, **_kwargs: session,
                    )

    def test_detect_uses_validated_size_for_tensor_and_output_mapping(self):
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, 0] = (300, 300, 340, 340, 0.9, 7)
        session = Session(
            inputs=[NodeArg("images", "tensor(float)", [1, 3, 640, 640])],
            result=output,
        )
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
        self.assertEqual(session.run_calls[0][1]["images"].shape, (1, 3, 640, 640))
        self.assertEqual(
            (detections[0].x1, detections[0].y1,
             detections[0].x2, detections[0].y2),
            (150.0, 150.0, 170.0, 170.0),
        )
```

Retain all current wrong-name, wrong-type, wrong-output, provider fallback,
malformed row, and bundled resource assertions.

- [ ] **Step 2: Run detector tests to verify RED**

Run:

```powershell
python -m unittest tests.test_ai_detection -v
```

Expected: failures because the current API has no `input_size`, rejects 160/640,
and does not resize or map coordinates.

- [ ] **Step 3: Implement exact supported-size validation and canonical mapping**

Refactor `ai_detection.py` around these exact interfaces:

```python
from image_resize import resize_rgb_bilinear


LOGICAL_FRAME_SIZE = 320
SUPPORTED_INPUT_SIZES = (160, 320, 640)
_INPUT_NAME = "images"
_OUTPUT_NAME = "output0"
_OUTPUT_SHAPE = [1, 300, 6]
_TENSOR_TYPE = "tensor(float)"


def _validated_input_size(raw: object) -> int:
    if type(raw) is not int or raw not in SUPPORTED_INPUT_SIZES:
        raise ModelContractError(
            "AI model input must be images tensor(float) "
            "[1,3,N,N] where N is 160, 320, or 640"
        )
    return raw


def preprocess_frame(
    frame: np.ndarray,
    input_size: int = LOGICAL_FRAME_SIZE,
) -> np.ndarray:
    if (
        not isinstance(frame, np.ndarray)
        or frame.shape != (LOGICAL_FRAME_SIZE, LOGICAL_FRAME_SIZE, 3)
    ):
        raise ValueError("AI frame must be RGB 320x320x3")
    if frame.dtype != np.uint8:
        raise ValueError("AI frame must use uint8 pixels")
    input_size = _validated_input_size(input_size)
    prepared = (
        frame
        if input_size == LOGICAL_FRAME_SIZE
        else resize_rgb_bilinear(frame, input_size)
    )
    return np.ascontiguousarray(
        prepared.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    )


def parse_output(
    output: np.ndarray,
    input_size: int = LOGICAL_FRAME_SIZE,
) -> tuple[Detection, ...]:
    input_size = _validated_input_size(input_size)
    if not isinstance(output, np.ndarray) or output.shape != tuple(_OUTPUT_SHAPE):
        raise ModelContractError(
            "AI model output must have shape [1, 300, 6]"
        )
    try:
        finite_rows = np.isfinite(output).all(axis=2)[0]
    except TypeError as error:
        raise ModelContractError(
            "AI model output must contain numeric values"
        ) from error

    scale = LOGICAL_FRAME_SIZE / input_size
    detections = []
    for row in output[0, finite_rows]:
        x1, y1, x2, y2, confidence, class_id = (
            float(value) for value in row
        )
        if not 0.0 <= confidence <= 1.0:
            continue
        if class_id < 0.0 or not class_id.is_integer():
            continue
        x1 = min(LOGICAL_FRAME_SIZE, max(0.0, x1 * scale))
        y1 = min(LOGICAL_FRAME_SIZE, max(0.0, y1 * scale))
        x2 = min(LOGICAL_FRAME_SIZE, max(0.0, x2 * scale))
        y2 = min(LOGICAL_FRAME_SIZE, max(0.0, y2 * scale))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(x1, y1, x2, y2, confidence, int(class_id))
        )
    return tuple(detections)
```

Make input validation return the size and expose it read-only:

```python
class OnnxDetector:
    @property
    def input_size(self) -> int:
        return self._input_size

    def _validate_contract(self) -> int:
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ModelContractError("AI model must have exactly one input")
        node = inputs[0]
        shape = list(node.shape)
        if (
            node.name != _INPUT_NAME
            or node.type != _TENSOR_TYPE
            or len(shape) != 4
            or shape[:2] != [1, 3]
            or shape[2] != shape[3]
        ):
            raise ModelContractError(
                "AI model input must be images tensor(float) "
                "[1,3,N,N] where N is 160, 320, or 640"
            )
        input_size = _validated_input_size(shape[2])
        self._validate_node(
            self._session.get_outputs(),
            _OUTPUT_NAME,
            _OUTPUT_SHAPE,
            "output",
        )
        return input_size

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        tensor = preprocess_frame(frame, self._input_size)
        output = self._session.run(
            [_OUTPUT_NAME], {_INPUT_NAME: tensor}
        )[0]
        return parse_output(output, self._input_size)
```

Assign `self._input_size = self._validate_contract()` before publishing provider
state. Do not inspect or infer size from the filename.

- [ ] **Step 4: Run detector, targeting, and zoom tests to verify GREEN**

Run:

```powershell
python -m unittest tests.test_ai_detection tests.test_ai_targeting tests.test_ai_zoom -v
```

Expected: all tests pass; target and zoom code continue receiving canonical
320-space detections.

- [ ] **Step 5: Commit multi-size detector support**

```powershell
git add ai_detection.py tests/test_ai_detection.py
git commit -m "feat: support 160 320 and 640 ONNX inputs"
```

---

### Task 3: Carry Validated Input Size Through Model Selection

**Files:**
- Modify: `ai_model_selection.py`
- Modify: `tests/test_ai_model_selection.py`

**Interfaces:**
- Consumes: read-only `OnnxDetector.input_size` from Task 2.
- Produces: `ModelChoice(path: Path, display_name: str, is_default: bool, input_size: int | None = None)`; ready events contain a validated choice with a supported size, while error events retain the original candidate and a safe actionable message.
- Preserves: all event-lock/generation semantics; `ModelValidationEvent` extends compatibly with `safe_message: str | None = None` after `error_type`.

- [ ] **Step 1: Add failing immutable metadata and validator enrichment tests**

Update `tests/test_ai_model_selection.py` with exact expectations:

```python
def test_bundled_choice_has_known_320_input_size(self):
    with mock.patch(
        "ai_model_selection.model_resource_path",
        return_value=Path("models/all_games_320.onnx"),
    ):
        choice = bundled_model_choice()
    self.assertEqual(choice.input_size, 320)


def test_external_choice_has_no_trusted_size_before_validation(self):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "custom.onnx"
        path.write_bytes(b"model")
        choice = external_model_choice(path)
    self.assertIsNone(choice.input_size)


def test_validator_publishes_choice_enriched_with_detector_input_size(self):
    events = []
    finished = threading.Event()

    class Detector:
        provider = "DmlExecutionProvider"
        input_size = 640

    choice = ModelChoice(Path("chosen.onnx"), "chosen.onnx", False)
    validator = ModelValidator(
        lambda event: (events.append(event), finished.set()),
        detector_factory=lambda _path: Detector(),
    )
    self.addCleanup(validator.close)
    self.assertTrue(validator.start(choice, 4))
    self.assertTrue(finished.wait(1.0))
    self.assertEqual(events[0].choice.input_size, 640)
    self.assertEqual(events[0].choice.path, choice.path)
    self.assertIsNone(choice.input_size)


def test_contract_failure_event_has_safe_actionable_message(self):
    events = []
    finished = threading.Event()

    def fail(_path):
        raise ModelContractError(
            "AI model input must use a 160, 320, or 640 square input"
        )

    choice = ModelChoice(Path("secret.onnx"), "secret.onnx", False)
    validator = ModelValidator(
        lambda event: (events.append(event), finished.set()),
        detector_factory=fail,
    )
    self.addCleanup(validator.close)
    with self.assertLogs("ai_model_selection", level="ERROR"):
        self.assertTrue(validator.start(choice, 8))
        self.assertTrue(finished.wait(1.0))
    self.assertEqual(events[0].error_type, "ModelContractError")
    self.assertEqual(
        events[0].safe_message,
        "AI model input must use a 160, 320, or 640 square input",
    )
```

Give every successful fake detector in existing validator tests an explicit
`input_size` of 160, 320, or 640, and update ready-event equality to expect the
enriched immutable choice. Keep failure-event expectations unchanged.

- [ ] **Step 2: Run model-selection tests to verify RED**

Run:

```powershell
python -m unittest tests.test_ai_model_selection -v
```

Expected: failures because `ModelChoice` has no `input_size` and ready events do
not enrich the choice.

- [ ] **Step 3: Implement immutable size metadata and ready-event enrichment**

Modify `ai_model_selection.py`:

```python
from dataclasses import dataclass, replace

from ai_detection import ModelContractError, OnnxDetector, model_resource_path


@dataclass(frozen=True)
class ModelChoice:
    path: Path
    display_name: str
    is_default: bool
    input_size: int | None = None


def bundled_model_choice() -> ModelChoice:
    path = model_resource_path().resolve()
    return ModelChoice(path, path.name, True, 320)


def external_model_choice(raw_path: str | Path) -> ModelChoice:
    path = Path(raw_path)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ModelSelectionError(
            "Selected model file was not found"
        ) from error
    if not resolved.is_file():
        raise ModelSelectionError("Selected model must be a file")
    if resolved.suffix.lower() != ".onnx":
        raise ModelSelectionError("Select an ONNX model file")
    return ModelChoice(resolved, resolved.name, False, None)


@dataclass(frozen=True)
class ModelValidationEvent:
    kind: str
    token: int
    choice: ModelChoice
    error_type: str | None = None
    safe_message: str | None = None
```

In `ModelValidator._worker`, construct the ready event only from validated
detector metadata:

```python
detector = self._detector_factory(choice.path)
_provider = detector.provider
validated_choice = replace(choice, input_size=detector.input_size)
event = ModelValidationEvent("ready", token, validated_choice)
```

For the error path, expose only contract messages authored by Jitter and keep
arbitrary exception details in the log:

```python
except Exception as error:
    LOGGER.exception("AI model validation failed for %s", choice.path)
    safe_message = (
        str(error)
        if isinstance(error, ModelContractError)
        else "AI model validation failed"
    )
    event = ModelValidationEvent(
        "error",
        token,
        choice,
        type(error).__name__,
        safe_message,
    )
```

Do not add filename parsing or trust a pre-filled external value. The service
will revalidate by constructing its own detector later.

- [ ] **Step 4: Run model-selection tests to verify GREEN**

Run:

```powershell
python -m unittest tests.test_ai_model_selection -v
```

Expected: all choice, thread, stale-token, cancellation, logging, and start
failure tests pass.

- [ ] **Step 5: Commit model validation metadata**

```powershell
git add ai_model_selection.py tests/test_ai_model_selection.py
git commit -m "feat: report validated ONNX input size"
```

---

### Task 4: Display and Commit Validated Model Size in the UI Lifecycle

**Files:**
- Modify: `ui.py:2940-3165`
- Modify: `tests/test_ui.py`

**Interfaces:**
- Consumes: enriched `ModelValidationEvent.choice` from Task 3.
- Produces: `_model_label(choice: ModelChoice) -> str` with a trusted size suffix; the active `_ModelSwitch.candidate` is replaced by the enriched ready choice before idle commit or runtime start; `_ModelSwitch.failure` carries a safe rollback footer message.
- Preserves: loading label without untrusted size, exact switch token checks, rollback-once, runtime generation checks, button availability, and no model persistence.

- [ ] **Step 1: Add failing UI label and enriched-switch tests**

Update the initial/browse expectations and add focused tests:

```python
def test_model_label_shows_only_validated_input_size(self):
    self.assertEqual(
        self.app._model_label(bundled_model_choice()),
        "Default · all_games_320.onnx · 320×320",
    )
    pending = ModelChoice(Path("custom.onnx"), "custom.onnx", False)
    validated = ModelChoice(Path("custom.onnx"), "custom.onnx", False, 640)
    self.assertEqual(self.app._model_label(pending), "Custom · custom.onnx")
    self.assertEqual(
        self.app._model_label(validated),
        "Custom · custom.onnx · 640×640",
    )


def test_ready_event_replaces_pending_candidate_with_validated_choice(self):
    pending, token = self.begin_custom_model_switch("custom.onnx")
    validated = replace(pending, input_size=160)
    self.model_validator.emit(ModelValidationEvent("ready", token, validated))
    self.drain_ui_queue()
    self.assertEqual(self.app._model_choice, validated)
    self.assertEqual(
        self.app.ai_model_var.get(),
        "Custom · custom.onnx · 160×160",
    )


def test_ready_event_with_same_token_but_different_path_is_ignored(self):
    pending, token = self.begin_custom_model_switch("expected.onnx")
    wrong = ModelChoice(Path("other.onnx"), "other.onnx", False, 640)
    self.model_validator.emit(ModelValidationEvent("ready", token, wrong))
    self.drain_ui_queue()
    self.assertEqual(self.app._model_switch.candidate, pending)
    self.assertEqual(
        self.app.ai_model_var.get(),
        "Loading · expected.onnx",
    )


def test_invalid_input_size_footer_is_actionable_without_path_leak(self):
    pending, token = self.begin_custom_model_switch("private-name.onnx")
    self.model_validator.emit(ModelValidationEvent(
        "error",
        token,
        pending,
        "ModelContractError",
        "AI model input must use a 160, 320, or 640 square input",
    ))
    self.drain_ui_queue()
    footer = self.app.footer_var.get()
    self.assertIn("160, 320, or 640", footer)
    self.assertNotIn(str(pending.path.parent), footer)
```

Where existing tests manually emit ready events, enrich the ready choice with a
supported size. Update committed default/custom label expectations to include
`· 320×320`, `· 160×160`, or `· 640×640`. Keep loading labels unchanged.

Add/retain a save regression proving neither external path nor size appears in
the saved configuration mapping.

- [ ] **Step 2: Run focused UI tests to verify RED**

Run:

```powershell
python -m unittest tests.test_ui -v
```

Expected: label and enriched-choice assertions fail against the old UI logic;
the old full-object equality check ignores the enriched ready choice.

- [ ] **Step 3: Implement validated size labels and path-safe event matching**

Change the label helper:

```python
@staticmethod
def _model_label(choice: ModelChoice) -> str:
    prefix = "Default" if choice.is_default else "Custom"
    label = f"{prefix} · {choice.display_name}"
    if choice.input_size is not None:
        label += f" · {choice.input_size}×{choice.input_size}"
    return label
```

In `handle_model_validation_event`, match the immutable switch identity by token
and path, then replace the switch candidate only for a ready event:

```python
switch = self._model_switch
if (
    switch is None
    or event.token != switch.token
    or event.choice.path != switch.candidate.path
):
    return
if event.kind == "error":
    self._start_model_rollback(
        switch, event.safe_message or "AI model validation failed"
    )
    return
if event.kind != "ready" or event.choice.input_size is None:
    return
validated_switch = replace(switch, candidate=event.choice)
self._model_switch = validated_switch
if not self._ai_runtime_required():
    self._finish_model_switch(
        event.choice,
        f"Using model: {event.choice.display_name}",
    )
    return
self._start_validated_model_generation(validated_switch)
```

Extend the private immutable switch state and retain the safe reason through an
active rollback generation:

```python
@dataclass(frozen=True)
class _ModelSwitch:
    token: int
    candidate: ModelChoice
    previous: ModelChoice
    phase: str
    failure: str | None = None


def _start_model_rollback(self, switch: _ModelSwitch, failure: str) -> None:
    if self._model_switch != switch:
        return
    logging.error(
        "AI model %s rejected: %s",
        switch.candidate.path,
        failure,
    )
    try:
        if self._ai_runtime_active:
            self._stop_ai_runtime("Model rollback")
    except Exception:
        logging.exception("AI model rollback cleanup failed")
    finally:
        self._ai_ready = False
        self._ai_provider = None
        self._ai_runtime_active = False
        self._sync_adaptive_zoom_gate()
    if not self._ai_runtime_required():
        self._finish_model_switch(
            switch.previous,
            f"Model rejected: {failure}; restored {switch.previous.display_name}",
        )
        return
    rollback = replace(
        switch,
        phase="starting_rollback",
        failure=failure,
    )
    self._model_switch = rollback
    self.ai_model_var.set(f"Loading · {rollback.previous.display_name}")
    self._render_model_controls()
    if self._start_ai_runtime(
        "Model rollback", model_choice=rollback.previous
    ):
        return
    self._handle_ai_runtime_error("AI model rollback failed")
```

When the rollback generation publishes its exact ready event, use
the same safe footer and keep full candidate paths only in the logging call:

```python
elif switch is not None and switch.phase == "starting_rollback":
    failure = switch.failure or "AI model validation failed"
    self._finish_model_switch(
        switch.previous,
        f"Model rejected: {failure}; restored {switch.previous.display_name}",
    )
```

Do not display size in `Loading · filename` because the file is not trusted until
validation completes. Rollback always uses the previous choice, which already
contains trusted metadata.

- [ ] **Step 4: Run complete model-selection and UI tests to verify GREEN**

Run:

```powershell
python -m unittest tests.test_ai_model_selection tests.test_ui -v
```

Expected: all UI layout/runtime, switch, cancellation, rollback, Test 3s,
configuration, and model metadata tests pass.

- [ ] **Step 5: Commit the UI lifecycle change**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: show validated AI model input size"
```

---

### Task 5: Update Distribution Inventories and User/Repository Documentation

**Files:**
- Modify: `tests/test_entrypoints.py`
- Modify: `AGENTS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the new top-level source `image_resize.py` and the finalized 160/320/640 runtime behavior.
- Produces: repository guidance and Thai README accurately describing fixed 320 capture and runtime-only multi-size models; entrypoint tests lock the published documentation contract.
- Preserves: exactly one packaged model data file and the bundled 320 hash/DirectML self-check.

- [ ] **Step 1: Add failing README contract assertions**

Add this focused assertion to `tests/test_entrypoints.py`:

```python
def test_readme_documents_supported_external_sizes_and_fixed_capture(self):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for contract in (
        "[1,3,160,160]",
        "[1,3,320,320]",
        "[1,3,640,640]",
        "พื้นที่ capture ยังคง 320×320",
        "128/256",
    ):
        with self.subTest(contract=contract):
            self.assertIn(contract, readme)
```

Require the entrypoint assertion to use an exact data-file option for
`models/all_games_320.onnx`, alongside only the approved licenses/sound
directories. Do not recursively package `models/` or add
`models/all_games_640.onnx`, `models/all_games.onnx`, or another model option.

- [ ] **Step 2: Run distribution and entrypoint tests to verify RED**

Run:

```powershell
python -m unittest tests.test_entrypoints -v
```

Expected: the new exact README strings are absent. All bundled
model/hash/data assertions remain green.

- [ ] **Step 3: Update AGENTS.md and the Thai README with exact behavior**

Add `image_resize.py` to the planned layout and replace the single external
input contract with:

```text
Browse... accepts only runtime external ONNX models whose `images` float input
is exactly [1,3,N,N] for N in 160, 320, or 640 and whose `output0` float output
is exactly [1,300,6]. Capture, Overlay, targeting, movement, and Adaptive Zoom
remain in canonical 320-by-320 coordinates; detector output is scaled back
before publication.
```

Document all of the following in `README.md`:

- auto-detected runtime sizes 160, 320, and 640;
- startup default stays bundled 320;
- labels display the validated size;
- 128, 256, dynamic, rectangular, and malformed models are rejected;
- capture/Overlay/FOV remain 320 for every model;
- 160 may require less inference work, 320 is the default balance, and 640 may
  require more work, without guaranteeing FPS or accuracy;
- 640 receives an upscale of the same physical 320 source area;
- external path and size are not persisted, copied, or packaged;
- self-check remains the bundled 320 hash and DirectML contract.

Update the README verification command to compile `image_resize.py`.

- [ ] **Step 4: Run documentation, distribution, and review checks to verify GREEN**

Run:

```powershell
python -m unittest tests.test_distribution_metadata tests.test_entrypoints -v
python .\distribution_metadata.py --review-json
git diff --check
```

Expected: tests pass; review JSON includes `image_resize.py` in
`compile_targets`, retains only approved data options, and reports the unchanged
bundled model configuration.

- [ ] **Step 5: Commit inventory and documentation updates**

```powershell
git add tests/test_entrypoints.py AGENTS.md README.md
git commit -m "docs: explain multi-size external ONNX support"
```

---

### Task 6: Complete Integration Verification and Review

**Files:**
- Verify only: all modified source, tests, and documentation from Tasks 1-5
- Preserve: untracked user model files and generated artifacts

**Interfaces:**
- Consumes: the complete feature branch.
- Produces: fresh verification evidence and a reviewed branch ready for the user's integration choice.

- [ ] **Step 1: Run focused multi-size regression suites**

```powershell
python -m unittest tests.test_image_resize tests.test_ai_detection tests.test_ai_model_selection tests.test_ai_zoom tests.test_ai_service tests.test_ai_targeting -v
```

Expected: all resize, model contract, canonical mapping, nearest-target,
same-frame zoom, service generation, and movement tests pass.

- [ ] **Step 2: Run exact repository compile verification**

```powershell
python -m py_compile main.py ui.py motion.py combined_motion.py image_resize.py ai_targeting.py ai_tracking.py ai_detection.py ai_capture.py ai_zoom.py ai_service.py ai_model_selection.py display_timing.py overlay.py makcu_service.py hotkeys.py settings.py sound_service.py liquid_widgets.py distribution_metadata.py
```

Expected: exit code 0 with no compiler output.

- [ ] **Step 3: Run the complete hardware-free test suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass with zero failures and zero errors. Error logs emitted
by deliberate failure-path tests do not count as failures; use the final unittest
summary and exit code.

- [ ] **Step 4: Run dependency, DirectML, distribution, and diff checks**

```powershell
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
git diff --check
```

Expected:

- imports exit 0;
- self-check JSON reports `"status": "ok"`, the approved 320 model hash, and
  `"provider": "DmlExecutionProvider"`;
- review JSON includes `image_resize.py` and only the exact
  `models/all_games_320.onnx` model data-file option;
- diff check emits no whitespace errors.

- [ ] **Step 5: Inspect the final repository state and protect user files**

```powershell
git status --short
git diff --stat main...HEAD
git log --oneline --decorate -8
```

Expected: only the known user-owned external model files remain untracked;
feature changes are committed, and no `build-output/`, config, log, or alternate
model file is staged or committed.

- [ ] **Step 6: Request a read-only code review**

Use `superpowers:requesting-code-review` against the branch fork point and HEAD.
The reviewer must specifically check:

- resize numerical stability and memory behavior at 640;
- exact static contract validation and rejection messages;
- scale-before-clip output mapping;
- trusted vs untrusted `ModelChoice.input_size` lifecycle;
- token/path generation safety for enriched ready events;
- unchanged rollback and cancellation behavior;
- no external model packaging or persistence;
- test coverage for every accepted/rejected size.

Resolve every Critical and Important finding with its own RED/GREEN regression
cycle, then repeat the affected focused suite and the complete verification.

- [ ] **Step 7: Perform connected-hardware verification when a Makcu and external models are available**

For one contract-compatible external model at each size 160, 320, and 640,
verify:

```text
Browse validation and displayed size
DirectML/CPU runtime readiness
Trigger and Modifier gating
Nearest head/player current-frame selection
Overlay alignment and Head Boxes visibility
Adaptive Zoom 1.0x, 1.5x, and 2.0x behavior
Jitter-only, AI-only, and combined movement
STOP, disconnect, model switch, rollback, Test 3s, and shutdown
Use Default returns to the bundled 320 model
```

If no compatible 160 model is available, report that the 160 hardware/runtime
check remains pending; do not copy, synthesize, rename, or package another model
as a substitute.

No commit is required for verification unless review findings require a code or
test change.
