# Raw Single-Class Ultralytics YOLO Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept runtime-browsed one-class Ultralytics ONNX detectors with exact raw output `[1,5,K]`, including `models/Apex_20k_pictures_640.onnx`, while preserving Jitter's legacy detector and canonical 320-space behavior.

**Architecture:** Add a pure NumPy decoder in `ai_yolo.py`; let `OnnxDetector` validate one of two exact output contracts once per generation and route each inference to the selected parser. All downstream code continues to receive immutable `Detection` tuples and stays format-independent.

**Tech Stack:** Python 3, NumPy, ONNX Runtime DirectML, `unittest`, existing Tkinter/Makcu application boundaries.

**Spec:** `docs/superpowers/specs/2026-08-29-raw-yolo-single-class-support-design.md`

## Global Constraints

- Keep `models/all_games_320.onnx` as the bundled startup default and preserve SHA-256 `6B9157D6419F9DBC40D2DCECCC33A3387078C86F1C5872EDA544B174FF48499C`.
- Continue accepting legacy `output0 tensor(float) [1,300,6]` models unchanged.
- Accept raw models only when input is exactly `images tensor(float) [1,3,N,N]`, where `N` is the static Python integer `160`, `320`, or `640`.
- Accept raw output only as `output0 tensor(float) [1,5,K]`, with `K=525`, `2100`, or `8400` for input `160`, `320`, or `640`, respectively.
- Require raw metadata `task == "detect"` and a safely parsed `names` dictionary containing exactly integer key `0` with one non-empty string label.
- Map the sole raw class to Jitter player class `0`; never synthesize class `7` head detections.
- Decode with NumPy only; do not add Torch, Ultralytics, OpenCV, Pillow, another runtime, or a new dependency.
- Use confidence floor `0.05`, NMS IoU threshold `0.45`, and a hard maximum of `300` published detections.
- Keep capture, Overlay, targeting, movement, and Adaptive Zoom in canonical `320x320` coordinates.
- Do not add UI controls or persisted settings for output format, NMS, labels, model paths, or external models.
- Never copy, rename, stage, commit, package, or persist `models/Apex_20k_pictures_640.onnx` or any other user-owned external model.
- Keep the canonical Nuitka data option exactly `--include-data-files=models/all_games_320.onnx=models/all_games_320.onnx`; do not run Nuitka for this feature.
- Preserve off-UI-thread validation, generation barriers, exact-ready switching, rollback-once, STOP/Test 3s cancellation, and AI failure isolation.
- Preserve unrelated user changes and never edit generated output, `build-output/`, `dist/`, `*.build/`, `*.dist/`, `__pycache__/`, or `app.log` as source.

## File Structure

- Create `ai_yolo.py`: pure raw single-class decode, stable bounded NMS, canonical coordinate mapping, and exact constants.
- Create `tests/test_ai_yolo.py`: exhaustive tensor, filtering, ordering, NMS, cap, and coordinate tests for the pure decoder.
- Modify `ai_detection.py`: distinguish the legacy and raw output contracts, validate raw metadata safely, and route runtime output to the fixed parser.
- Modify `tests/test_ai_detection.py`: fake model metadata, contract recognition/rejection, parser routing, runtime shape checks, provider preservation, and downstream integration.
- Modify `tests/test_ai_model_selection.py`: verify raw-contract failures remain concise in the UI event while diagnostics stay in logs.
- Modify `tests/test_distribution_metadata.py` and `tests/test_entrypoints.py`: add `ai_yolo.py` to the explicit reviewed compile inventory while continuing to assert that only the bundled model is packaged.
- Modify `README.md`: document both exact external output families, metadata rules, one-class mapping, and limitations in Thai.
- Modify `AGENTS.md`: keep repository guidance synchronized with the approved dual-contract implementation and verification command.

---

### Task 1: Pure raw YOLO decoder and bounded deterministic NMS

**Files:**
- Create: `ai_yolo.py`
- Create: `tests/test_ai_yolo.py`
- Modify: `tests/test_distribution_metadata.py:264-282`
- Modify: `tests/test_entrypoints.py:319-343`

**Interfaces:**
- Consumes: `ai_targeting.Detection` and a numeric NumPy array with exact shape `[1,5,K]`.
- Produces: `MIN_CONFIDENCE = 0.05`, `NMS_IOU_THRESHOLD = 0.45`, `MAX_DETECTIONS = 300`, `RAW_CANDIDATE_COUNTS = {160: 525, 320: 2100, 640: 8400}`, and `decode_single_class_yolo(output: np.ndarray, input_size: int) -> tuple[Detection, ...]`.
- Private helper contract: `_nms_keep_positions(boxes: np.ndarray, confidences: np.ndarray, raw_indices: np.ndarray) -> np.ndarray`; inputs have shapes `(M,4)`, `(M,)`, `(M,)`, and the returned candidate positions are ordered by original raw index.

- [ ] **Step 1: Write failing decoder contract and geometry tests**

Create `tests/test_ai_yolo.py` with imports and helpers that make exact raw tensors without a model fixture:

```python
import unittest

import numpy as np

from ai_targeting import Detection
from ai_yolo import (
    MAX_DETECTIONS,
    MIN_CONFIDENCE,
    NMS_IOU_THRESHOLD,
    RAW_CANDIDATE_COUNTS,
    _nms_keep_positions,
    decode_single_class_yolo,
)


def raw_output(input_size: int) -> np.ndarray:
    return np.zeros(
        (1, 5, RAW_CANDIDATE_COUNTS[input_size]), dtype=np.float32
    )


def put_candidate(
    output: np.ndarray,
    index: int,
    cx: float,
    cy: float,
    width: float,
    height: float,
    confidence: float,
) -> None:
    output[0, :, index] = (cx, cy, width, height, confidence)


class RawYoloDecoderTests(unittest.TestCase):
    def test_contract_constants_are_exact(self):
        self.assertEqual(MIN_CONFIDENCE, 0.05)
        self.assertEqual(NMS_IOU_THRESHOLD, 0.45)
        self.assertEqual(MAX_DETECTIONS, 300)
        self.assertEqual(
            RAW_CANDIDATE_COUNTS, {160: 525, 320: 2100, 640: 8400}
        )

    def test_center_boxes_scale_to_canonical_space_for_every_input_size(self):
        for input_size in (160, 320, 640):
            with self.subTest(input_size=input_size):
                output = raw_output(input_size)
                put_candidate(output, 0, input_size / 2, input_size / 2,
                              input_size / 4, input_size / 2, 0.80)
                detection = decode_single_class_yolo(output, input_size)[0]
                self.assertEqual(
                    (detection.x1, detection.y1, detection.x2, detection.y2,
                     detection.class_id),
                    (120.0, 80.0, 200.0, 240.0, 0),
                )
                self.assertAlmostEqual(detection.confidence, 0.8, places=6)

    def test_coordinates_clip_after_scaling_and_collapsed_boxes_are_removed(self):
        for input_size in (160, 320, 640):
            with self.subTest(input_size=input_size):
                output = raw_output(input_size)
                put_candidate(
                    output, 0,
                    input_size * 0.05, input_size * 0.05,
                    input_size * 0.20, input_size * 0.20, 0.90,
                )
                put_candidate(
                    output, 1,
                    -input_size * 0.10, input_size * 0.05,
                    input_size * 0.05, input_size * 0.05, 0.95,
                )
                detections = decode_single_class_yolo(output, input_size)
                self.assertEqual(len(detections), 1)
                np.testing.assert_allclose(
                    (detections[0].x1, detections[0].y1,
                     detections[0].x2, detections[0].y2),
                    (0.0, 0.0, 48.0, 48.0),
                    atol=1e-5,
                )

    def test_invalid_rows_are_skipped_without_mutating_output(self):
        output = raw_output(320)
        candidates = (
            (100, 100, 20, 20, 0.80),
            (100, 100, 20, 20, 0.049),
            (100, 100, 20, 20, 1.01),
            (100, 100, 0, 20, 0.90),
            (100, 100, -1, 20, 0.90),
            (np.nan, 100, 20, 20, 0.90),
            (100, np.inf, 20, 20, 0.90),
        )
        for index, values in enumerate(candidates):
            put_candidate(output, index, *values)
        before = output.copy()
        detections = decode_single_class_yolo(output, 320)
        np.testing.assert_array_equal(output, before)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, 0)

    def test_exact_confidence_floor_is_inclusive(self):
        output = raw_output(160)
        put_candidate(output, 0, 20, 20, 10, 10, 0.05)
        self.assertEqual(len(decode_single_class_yolo(output, 160)), 1)
        self.assertEqual(decode_single_class_yolo(raw_output(160), 160), ())

    def test_rejects_wrong_shape_orientation_size_and_nonnumeric_array(self):
        invalid = (
            (np.zeros((1, 5, 524), dtype=np.float32), 160),
            (np.zeros((1, 525, 5), dtype=np.float32), 160),
            (np.zeros((1, 5, 525), dtype=np.float32), 320),
            (np.zeros((1, 5, 525), dtype="U1"), 160),
            (np.zeros((1, 5, 525), dtype=np.bool_), 160),
        )
        for output, input_size in invalid:
            with self.subTest(shape=output.shape, input_size=input_size):
                with self.assertRaises(ValueError):
                    decode_single_class_yolo(output, input_size)
        with self.assertRaises(ValueError):
            decode_single_class_yolo([[[]]], 160)
        with self.assertRaises(ValueError):
            decode_single_class_yolo(raw_output(160), 128)
        with self.assertRaises(ValueError):
            decode_single_class_yolo(raw_output(160), 160.0)
        with self.assertRaises(ValueError):
            decode_single_class_yolo(raw_output(160), True)
```

- [ ] **Step 2: Write failing NMS ordering, boundary, and hard-cap tests**

Append these tests to `RawYoloDecoderTests`:

```python
    def test_nms_suppresses_only_iou_strictly_greater_than_threshold(self):
        boxes = np.asarray(
            [[0, 0, 29, 10], [11, 0, 40, 10], [10, 0, 39, 10]],
            dtype=np.float64,
        )
        confidences = np.asarray([0.9, 0.8, 0.7], dtype=np.float64)
        raw_indices = np.asarray([0, 1, 2], dtype=np.int64)
        kept = _nms_keep_positions(boxes, confidences, raw_indices)
        np.testing.assert_array_equal(kept, [0, 1])

    def test_equal_confidence_nms_and_final_output_preserve_detector_order(self):
        output = raw_output(320)
        put_candidate(output, 8, 100, 100, 40, 40, 0.90)
        put_candidate(output, 3, 102, 102, 40, 40, 0.90)
        put_candidate(output, 5, 250, 250, 20, 20, 0.95)
        detections = decode_single_class_yolo(output, 320)
        self.assertEqual(
            [(item.x1, item.y1) for item in detections],
            [(82.0, 82.0), (240.0, 240.0)],
        )
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=6)
        self.assertAlmostEqual(detections[1].confidence, 0.95, places=6)

    def test_decoder_stops_after_three_hundred_nonoverlapping_survivors(self):
        output = raw_output(160)
        for index in range(301):
            put_candidate(
                output, index, 0.1 + index * 0.4, 1.0, 0.1, 0.1, 0.90
            )
        detections = decode_single_class_yolo(output, 160)
        self.assertEqual(len(detections), MAX_DETECTIONS)
        self.assertAlmostEqual(detections[0].x1, 0.1, places=6)
        self.assertLess(detections[-1].x1, 240.0)

    def test_private_nms_interface_uses_one_to_many_inputs_not_pairwise_iou(self):
        boxes = np.asarray([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float64)
        kept = _nms_keep_positions(
            boxes,
            np.asarray([0.8, 0.7], dtype=np.float64),
            np.asarray([4, 9], dtype=np.int64),
        )
        self.assertEqual(boxes.shape, (2, 4))
        np.testing.assert_array_equal(kept, [0, 1])
```

- [ ] **Step 3: Run the new test module and verify the RED state**

Run: `python -m unittest tests.test_ai_yolo -v`

Expected: FAIL during import with `ModuleNotFoundError: No module named 'ai_yolo'`.

- [ ] **Step 4: Implement the pure decoder minimally**

Create `ai_yolo.py` with this module boundary and algorithm:

```python
"""Pure NumPy post-processing for exact raw one-class YOLO outputs."""

import numpy as np

from ai_targeting import Detection


LOGICAL_FRAME_SIZE = 320
MIN_CONFIDENCE = 0.05
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 300
RAW_CANDIDATE_COUNTS = {160: 525, 320: 2100, 640: 8400}


def _nms_keep_positions(
    boxes: np.ndarray,
    confidences: np.ndarray,
    raw_indices: np.ndarray,
) -> np.ndarray:
    order = np.argsort(-confidences, kind="stable")
    kept: list[int] = []
    while order.size and len(kept) < MAX_DETECTIONS:
        current = int(order[0])
        kept.append(current)
        remaining = order[1:]
        if not remaining.size:
            break
        others = boxes[remaining]
        left = np.maximum(boxes[current, 0], others[:, 0])
        top = np.maximum(boxes[current, 1], others[:, 1])
        right = np.minimum(boxes[current, 2], others[:, 2])
        bottom = np.minimum(boxes[current, 3], others[:, 3])
        intersection = np.maximum(0.0, right - left) * np.maximum(
            0.0, bottom - top
        )
        current_area = (
            (boxes[current, 2] - boxes[current, 0])
            * (boxes[current, 3] - boxes[current, 1])
        )
        other_areas = (
            (others[:, 2] - others[:, 0])
            * (others[:, 3] - others[:, 1])
        )
        union = current_area + other_areas - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        )
        order = remaining[iou <= NMS_IOU_THRESHOLD]
    kept_array = np.asarray(kept, dtype=np.int64)
    if not kept_array.size:
        return kept_array
    return kept_array[np.argsort(raw_indices[kept_array], kind="stable")]


def decode_single_class_yolo(
    output: np.ndarray,
    input_size: int,
) -> tuple[Detection, ...]:
    if type(input_size) is not int:
        raise ValueError("Raw YOLO input size must be 160, 320, or 640")
    expected_candidates = RAW_CANDIDATE_COUNTS.get(input_size)
    expected_shape = (1, 5, expected_candidates) if expected_candidates else None
    if (
        not isinstance(output, np.ndarray)
        or output.shape != expected_shape
        or not np.isrealobj(output)
        or not (
            np.issubdtype(output.dtype, np.integer)
            or np.issubdtype(output.dtype, np.floating)
        )
    ):
        raise ValueError(
            "Raw YOLO output must be a numeric [1,5,K] tensor matching "
            "input size 160, 320, or 640"
        )

    rows = output[0].T.astype(np.float64, copy=False)
    finite = np.isfinite(rows).all(axis=1)
    confidence = rows[:, 4]
    valid = (
        finite
        & (confidence >= MIN_CONFIDENCE)
        & (confidence <= 1.0)
        & (rows[:, 2] > 0.0)
        & (rows[:, 3] > 0.0)
    )
    raw_indices = np.flatnonzero(valid)
    if not raw_indices.size:
        return ()
    rows = rows[raw_indices]
    boxes = np.column_stack((
        rows[:, 0] - rows[:, 2] / 2.0,
        rows[:, 1] - rows[:, 3] / 2.0,
        rows[:, 0] + rows[:, 2] / 2.0,
        rows[:, 1] + rows[:, 3] / 2.0,
    ))
    finite_boxes = np.isfinite(boxes).all(axis=1)
    boxes = boxes[finite_boxes]
    confidences = rows[finite_boxes, 4]
    raw_indices = raw_indices[finite_boxes]
    if not raw_indices.size:
        return ()
    kept = _nms_keep_positions(boxes, confidences, raw_indices)
    scale = LOGICAL_FRAME_SIZE / input_size
    detections = []
    for position in kept:
        x1, y1, x2, y2 = np.clip(
            boxes[position] * scale, 0.0, LOGICAL_FRAME_SIZE
        )
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(
            float(x1), float(y1), float(x2), float(y2),
            float(confidences[position]), 0,
        ))
    return tuple(detections)
```

Before accepting the implementation, check the source directly: the suppression loop must be bounded by `MAX_DETECTIONS` and must never construct a `(K,K)` array.

- [ ] **Step 5: Add the new source module to explicit reviewed inventories**

In both explicit expected compile-target sets, insert `"ai_yolo.py"` adjacent to `"ai_detection.py"`:

```python
"ai_targeting.py", "ai_tracking.py", "ai_detection.py", "ai_yolo.py",
```

Do not add a model data option. Keep the expected model option exactly:

```python
"--include-data-files=models/all_games_320.onnx=models/all_games_320.onnx"
```

- [ ] **Step 6: Run focused decoder and distribution tests**

Run:

```powershell
python -m unittest tests.test_ai_yolo -v
python -m unittest tests.test_distribution_metadata tests.test_entrypoints -v
```

Expected: all tests PASS; review payload contains `ai_yolo.py`, and its model data options contain only the existing bundled model plus existing license/sound directories.

- [ ] **Step 7: Commit the independently reviewable decoder**

```powershell
git add ai_yolo.py tests/test_ai_yolo.py tests/test_distribution_metadata.py tests/test_entrypoints.py
git commit -m "feat: decode raw single-class YOLO output"
```

Confirm the user-owned model remains untracked with `git status --short`.

---

### Task 2: Dual detector contract, safe metadata validation, and format-independent publication

**Files:**
- Modify: `ai_detection.py:1-132`
- Modify: `tests/test_ai_detection.py:1-291`
- Modify: `tests/test_ai_model_selection.py:245-286`

**Interfaces:**
- Consumes: `ai_yolo.RAW_CANDIDATE_COUNTS` and `ai_yolo.decode_single_class_yolo(output, input_size)`.
- Produces: `OnnxDetector.input_size: int` and `OnnxDetector.provider: str` unchanged; private `_output_format` is exactly `"post_nms"` or `"raw_single_class"` and is fixed at construction.
- Model metadata boundary: `session.get_modelmeta().custom_metadata_map` must be a string-key/string-value mapping with exact raw fields `task` and `names`; `names` is parsed only with `ast.literal_eval`.

- [ ] **Step 1: Extend detector fakes without changing legacy defaults**

Add metadata support near `NodeArg` in `tests/test_ai_detection.py`:

```python
class ModelMeta:
    def __init__(self, values=None):
        self.custom_metadata_map = values or {}


class Session:
    def __init__(
        self, inputs=None, outputs=None, providers=None, result=None,
        metadata=None,
    ):
        self._inputs = inputs or [
            NodeArg("images", "tensor(float)", [1, 3, 320, 320])
        ]
        self._outputs = outputs or [
            NodeArg("output0", "tensor(float)", [1, 300, 6])
        ]
        self._providers = providers or [
            "DmlExecutionProvider", "CPUExecutionProvider"
        ]
        self._metadata = ModelMeta(metadata)
        self.result = result if result is not None else valid_output()
        self.run_calls = []

    def get_modelmeta(self):
        return self._metadata
```

Keep existing methods and the legacy `valid_output()` helper unchanged.

- [ ] **Step 2: Write failing exact raw-contract acceptance and routing tests**

Add these helpers and tests to `tests/test_ai_detection.py`:

```python
RAW_COUNTS = {160: 525, 320: 2100, 640: 8400}
RAW_METADATA = {"task": "detect", "names": "{0: 'Enemy'}"}


def valid_raw_output(input_size=640):
    output = np.zeros((1, 5, RAW_COUNTS[input_size]), dtype=np.float32)
    output[0, :, 0] = (320, 320, 80, 160, 0.90)
    return output


class OnnxDetectorTests(unittest.TestCase):
    def test_accepts_exact_raw_contract_for_each_supported_input_size(self):
        for input_size, candidate_count in RAW_COUNTS.items():
            with self.subTest(input_size=input_size):
                session = Session(
                    inputs=[NodeArg(
                        "images", "tensor(float)",
                        [1, 3, input_size, input_size],
                    )],
                    outputs=[NodeArg(
                        "output0", "tensor(float)", [1, 5, candidate_count]
                    )],
                    result=valid_raw_output(input_size),
                    metadata=RAW_METADATA,
                )
                detector = OnnxDetector(
                    "model.onnx",
                    session_factory=lambda *_args, **_kwargs: session,
                )
                self.assertEqual(detector.input_size, input_size)

    def test_raw_detect_routes_to_decoder_and_maps_enemy_to_player(self):
        session = Session(
            inputs=[NodeArg("images", "tensor(float)", [1, 3, 640, 640])],
            outputs=[NodeArg("output0", "tensor(float)", [1, 5, 8400])],
            result=valid_raw_output(640),
            metadata=RAW_METADATA,
        )
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
        self.assertEqual(len(detections), 1)
        self.assertEqual(
            (detections[0].x1, detections[0].y1,
             detections[0].x2, detections[0].y2,
             detections[0].class_id),
            (140.0, 120.0, 180.0, 200.0, 0),
        )
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=6)
        self.assertEqual(session.run_calls[0][1]["images"].shape,
                         (1, 3, 640, 640))

    def test_legacy_contract_does_not_require_ultralytics_metadata(self):
        session = Session(metadata={})
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        self.assertEqual(len(detector.detect(
            np.zeros((320, 320, 3), dtype=np.uint8)
        )), 1)

    def test_raw_single_class_label_is_informational(self):
        for label in ("Enemy", "enemy", "person", "custom target"):
            with self.subTest(label=label):
                session = Session(
                    outputs=[NodeArg(
                        "output0", "tensor(float)", [1, 5, 2100]
                    )],
                    result=valid_raw_output(320),
                    metadata={"task": "detect", "names": repr({0: label})},
                )
                detector = OnnxDetector(
                    "model.onnx",
                    session_factory=lambda *_args, **_kwargs: session,
                )
                self.assertEqual(detector.detect(
                    np.zeros((320, 320, 3), dtype=np.uint8)
                )[0].class_id, 0)
```

Add `Detection` to the test imports from `ai_targeting`.

- [ ] **Step 3: Write failing contract-rejection and safe-metadata tests**

Add tests that cover every rejected raw family without evaluating metadata:

```python
    def test_rejects_raw_shape_mismatch_orientation_and_wrong_static_types(self):
        rejected = (
            (160, [1, 5, 2100]),
            (320, [1, 2100, 5]),
            (640, [1, 6, 8400]),
            (640, [1, 5, 8399]),
            (640, [1, 5, 8400.0]),
        )
        for input_size, output_shape in rejected:
            with self.subTest(input_size=input_size, output_shape=output_shape):
                session = Session(
                    inputs=[NodeArg(
                        "images", "tensor(float)",
                        [1, 3, input_size, input_size],
                    )],
                    outputs=[NodeArg(
                        "output0", "tensor(float)", output_shape
                    )],
                    metadata=RAW_METADATA,
                )
                with self.assertRaisesRegex(
                    ModelContractError, "\\[1,300,6\\].*raw one-class"
                ):
                    OnnxDetector(
                        "model.onnx",
                        session_factory=lambda *_args, **_kwargs: session,
                    )

    def test_raw_contract_requires_safe_exact_one_class_detect_metadata(self):
        rejected_metadata = (
            {},
            {"task": "segment", "names": "{0: 'Enemy'}"},
            {"task": "detect", "names": ""},
            {"task": "detect", "names": "not a literal"},
            {"task": "detect", "names": "__import__('os').system('bad')"},
            {"task": "detect", "names": "{False: 'Enemy'}"},
            {"task": "detect", "names": "{1: 'Enemy'}"},
            {"task": "detect", "names": "{0: ''}"},
            {"task": "detect", "names": "{0: 'Enemy', 1: 'Other'}"},
        )
        for metadata in rejected_metadata:
            with self.subTest(metadata=metadata):
                session = Session(
                    outputs=[NodeArg(
                        "output0", "tensor(float)", [1, 5, 2100]
                    )],
                    result=valid_raw_output(320),
                    metadata=metadata,
                )
                with self.assertRaisesRegex(
                    ModelContractError, "metadata.*one named class 0"
                ):
                    OnnxDetector(
                        "model.onnx",
                        session_factory=lambda *_args, **_kwargs: session,
                    )

    def test_raw_runtime_output_shape_is_rechecked(self):
        session = Session(
            outputs=[NodeArg("output0", "tensor(float)", [1, 5, 2100])],
            result=np.zeros((1, 2100, 5), dtype=np.float32),
            metadata=RAW_METADATA,
        )
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        with self.assertRaisesRegex(
            ModelContractError, "\\[1,300,6\\].*raw one-class"
        ):
            detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
```

Also retain the existing legacy runtime-shape test and DML-first/CPU-fallback tests without changing their assertions.

- [ ] **Step 4: Write a failing detector-to-downstream integration test**

Import `AimSettings` and `analyze_detections` from `ai_targeting`, `ZoomTransform` and `compose_zoom_refinement` from `ai_zoom`, `project_overlay_boxes` from `overlay`, and add:

```python
    def test_raw_players_flow_to_nearest_target_and_head_hidden_overlay(self):
        output = np.zeros((1, 5, 2100), dtype=np.float32)
        output[0, :, 0] = (40, 40, 20, 40, 0.95)
        output[0, :, 1] = (160, 170, 40, 100, 0.80)
        session = Session(
            outputs=[NodeArg("output0", "tensor(float)", [1, 5, 2100])],
            result=output,
            metadata=RAW_METADATA,
        )
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
        analysis = analyze_detections(
            detections, AimSettings(target_area="head"),
            sequence=1, captured_at=10.0,
        )
        self.assertEqual(analysis.frame.selected_index, 1)
        self.assertEqual(analysis.target.target_class, "player")
        self.assertEqual(
            (analysis.target.aim_x, analysis.target.aim_y), (160.0, 140.0)
        )
        self.assertEqual(
            len(project_overlay_boxes(
                analysis.frame, now=10.0, show_heads=False
            )),
            2,
        )

    def test_raw_player_refinement_stays_in_canonical_zoom_geometry(self):
        base_output = np.zeros((1, 5, 2100), dtype=np.float32)
        base_output[0, :, 0] = (160, 170, 40, 100, 0.90)
        refined_output = np.zeros((1, 5, 2100), dtype=np.float32)
        refined_output[0, :, 0] = (160, 160, 40, 160, 0.95)
        session = Session(
            outputs=[NodeArg("output0", "tensor(float)", [1, 5, 2100])],
            result=base_output,
            metadata=RAW_METADATA,
        )
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        base = analyze_detections(
            detector.detect(frame), AimSettings(target_area="head"),
            sequence=4, captured_at=20.0,
        )
        session.result = refined_output
        refined = compose_zoom_refinement(
            base,
            detector.detect(frame),
            ZoomTransform(80, 80, 160, 2.0),
            AimSettings(target_area="head"),
        )
        self.assertIsNotNone(refined)
        self.assertEqual(
            (refined.frame.detections[0].x1,
             refined.frame.detections[0].y1,
             refined.frame.detections[0].x2,
             refined.frame.detections[0].y2,
             refined.frame.detections[0].class_id),
            (150.0, 120.0, 170.0, 200.0, 0),
        )
```

These tests prove the raw model uses the existing player head-fallback aim point, competes by current-frame crosshair distance, remains visible when only head boxes are hidden, and passes through same-frame Adaptive Zoom in canonical coordinates.

- [ ] **Step 5: Extend model-selection safe-message coverage**

In `tests/test_ai_model_selection.py`, add a validator test using the exact authored detector failure:

```python
    def test_dual_output_contract_failure_is_safe_and_actionable(self):
        events = []
        finished = threading.Event()
        message = (
            "AI model output must be output0 tensor(float) [1,300,6] "
            "or supported raw one-class [1,5,K]"
        )

        def fail(_path):
            raise ModelContractError(message)

        choice = ModelChoice(Path("private-model.onnx"), "private-model.onnx", False)
        validator = ModelValidator(
            lambda event: (events.append(event), finished.set()),
            detector_factory=fail,
        )
        self.addCleanup(validator.close)
        with self.assertLogs("ai_model_selection", level="ERROR"):
            self.assertTrue(validator.start(choice, 19))
            self.assertTrue(finished.wait(1.0))
        self.assertEqual(events[0].safe_message, message)
        self.assertNotIn(str(choice.path.resolve()), events[0].safe_message)
```

- [ ] **Step 6: Run detector tests and verify the RED state**

Run:

```powershell
python -m unittest tests.test_ai_detection tests.test_ai_model_selection -v
```

Expected: raw acceptance/routing tests FAIL because `OnnxDetector` still accepts only `[1,300,6]`; all legacy tests remain green.

- [ ] **Step 7: Implement exact output-format and metadata validation**

In `ai_detection.py`:

1. Update the module docstring to describe exact dual-contract ONNX detection.
2. Import `ast`, `Mapping` from `collections.abc`, and the decoder symbols:

```python
import ast
from collections.abc import Mapping

from ai_yolo import RAW_CANDIDATE_COUNTS, decode_single_class_yolo
```

3. Rename `_OUTPUT_SHAPE` to `_POST_NMS_OUTPUT_SHAPE` and define:

```python
_POST_NMS_OUTPUT_SHAPE = [1, 300, 6]
_POST_NMS_FORMAT = "post_nms"
_RAW_SINGLE_CLASS_FORMAT = "raw_single_class"
_OUTPUT_CONTRACT_MESSAGE = (
    "AI model output must be output0 tensor(float) [1,300,6] "
    "or supported raw one-class [1,5,K]"
)
_RAW_METADATA_MESSAGE = (
    "Raw YOLO metadata must declare task=detect and exactly one named class 0"
)
```

4. Add safe metadata validation:

```python
def _validate_raw_metadata(session: object) -> None:
    try:
        metadata = session.get_modelmeta().custom_metadata_map
    except Exception as error:
        raise ModelContractError(_RAW_METADATA_MESSAGE) from error
    if not isinstance(metadata, Mapping) or metadata.get("task") != "detect":
        raise ModelContractError(_RAW_METADATA_MESSAGE)
    raw_names = metadata.get("names")
    if not isinstance(raw_names, str):
        raise ModelContractError(_RAW_METADATA_MESSAGE)
    try:
        names = ast.literal_eval(raw_names)
    except (SyntaxError, ValueError, TypeError) as error:
        raise ModelContractError(_RAW_METADATA_MESSAGE) from error
    if not isinstance(names, dict) or len(names) != 1:
        raise ModelContractError(_RAW_METADATA_MESSAGE)
    key, label = next(iter(names.items()))
    if type(key) is not int or key != 0 or not isinstance(label, str) or not label:
        raise ModelContractError(_RAW_METADATA_MESSAGE)
```

5. Make `_validate_contract()` return `(input_size, output_format)`, require one exact output name/type/static shape, and recognize formats only by exact shape:

```python
def _validate_contract(self) -> tuple[int, str]:
    inputs = self._session.get_inputs()
    if len(inputs) != 1:
        raise ModelContractError("AI model must have exactly one input")
    node = inputs[0]
    shape = list(node.shape)
    if (
        node.name != _INPUT_NAME
        or node.type != _TENSOR_TYPE
        or len(shape) != 4
        or any(type(dimension) is not int for dimension in shape)
        or shape[:2] != [1, 3]
        or shape[2] != shape[3]
    ):
        raise ModelContractError(
            "AI model input must be images tensor(float) "
            "[1,3,N,N] where N is 160, 320, or 640"
        )
    input_size = _validated_input_size(shape[2])
    return input_size, self._validate_output_contract(input_size)


def _validate_output_contract(self, input_size: int) -> str:
    outputs = self._session.get_outputs()
    if len(outputs) != 1:
        raise ModelContractError("AI model must have exactly one output")
    node = outputs[0]
    shape = list(node.shape)
    if (
        node.name != _OUTPUT_NAME
        or node.type != _TENSOR_TYPE
        or any(type(dimension) is not int for dimension in shape)
    ):
        raise ModelContractError(_OUTPUT_CONTRACT_MESSAGE)
    if shape == _POST_NMS_OUTPUT_SHAPE:
        return _POST_NMS_FORMAT
    if shape == [1, 5, RAW_CANDIDATE_COUNTS[input_size]]:
        _validate_raw_metadata(self._session)
        return _RAW_SINGLE_CLASS_FORMAT
    raise ModelContractError(_OUTPUT_CONTRACT_MESSAGE)
```

Replace every remaining legacy `_OUTPUT_SHAPE` reference with
`_POST_NMS_OUTPUT_SHAPE`; keep `parse_output()` behavior and its authored
legacy runtime-shape error unchanged.

Assign both immutable construction results:

```python
self._input_size, self._output_format = self._validate_contract()
```

6. Route runtime output through the chosen parser and wrap either parser's runtime contract error with the same safe dual-family message:

```python
def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
    tensor = preprocess_frame(frame, self._input_size)
    output = self._session.run([_OUTPUT_NAME], {_INPUT_NAME: tensor})[0]
    try:
        if self._output_format == _POST_NMS_FORMAT:
            return parse_output(output, self._input_size)
        return decode_single_class_yolo(output, self._input_size)
    except (ModelContractError, TypeError, ValueError) as error:
        raise ModelContractError(_OUTPUT_CONTRACT_MESSAGE) from error
```

Do not inspect filenames, expose `_output_format` in the UI, or alter provider fallback.

- [ ] **Step 8: Run detector, targeting, zoom, overlay, and lifecycle regressions**

Run:

```powershell
python -m unittest tests.test_ai_detection tests.test_ai_model_selection -v
python -m unittest tests.test_ai_targeting tests.test_ai_zoom tests.test_overlay -v
python -m unittest tests.test_ai_service -v
python -m unittest tests.test_combined_motion tests.test_ui -v
```

Expected: all PASS. In particular, the existing service cases for stale generations, no-target publication, nearest-current-frame selection, refinement fallback, rollback, STOP, and Test 3s remain unchanged because those modules never branch on output format.

- [ ] **Step 9: Commit the detector integration**

```powershell
git add ai_detection.py tests/test_ai_detection.py tests/test_ai_model_selection.py
git commit -m "feat: accept exact raw one-class ONNX models"
```

Confirm `git diff HEAD^ -- ai_service.py ai_targeting.py ai_zoom.py overlay.py ui.py settings.py` is empty.

---

### Task 3: Repository contract and Thai user documentation

**Files:**
- Modify: `README.md:236-291`
- Modify: `README.md:407`
- Modify: `AGENTS.md:31-34`
- Modify: `AGENTS.md:96-112`
- Modify: `AGENTS.md:237`
- Modify: `tests/test_entrypoints.py:499-540`

**Interfaces:**
- Consumes: the exact contracts and constants implemented in Tasks 1 and 2.
- Produces: Thai runtime-selection documentation, synchronized repository guidance, and a test-enforced statement that external raw models remain runtime-only and unpackaged.

- [ ] **Step 1: Write a failing documentation contract test**

Add to `tests/test_entrypoints.py`:

```python
    def test_readme_documents_exact_raw_single_class_contract_without_packaging_it(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("[1,5,K]", readme)
        self.assertIn("160 → K=525", readme)
        self.assertIn("320 → K=2100", readme)
        self.assertIn("640 → K=8400", readme)
        self.assertIn("task=detect", readme)
        self.assertIn("class 0", readme)
        self.assertIn("ai_yolo.py", readme)
        self.assertNotIn(
            "models/Apex_20k_pictures_640.onnx=models/Apex_20k_pictures_640.onnx",
            readme,
        )
```

- [ ] **Step 2: Run the documentation test and verify the RED state**

Run: `python -m unittest tests.test_entrypoints.EntryPointTests.test_readme_documents_exact_raw_single_class_contract_without_packaging_it -v`

Expected: FAIL because README currently documents only `[1,300,6]` and its compile command omits `ai_yolo.py`.

- [ ] **Step 3: Update the Thai ONNX model-selection documentation**

Replace the single-output-contract text/table in `README.md` with a Thai explanation containing these exact technical facts:

```markdown
รองรับ output ภายนอก 2 แบบ โดยระบบตรวจจาก contract ของโมเดลอัตโนมัติ:

1. แบบ post-NMS เดิม: `output0` ชนิด float รูปร่าง `[1,300,6]`
   (`x1,y1,x2,y2,confidence,class_id`) โดย class `0` คือ player และ class `7` คือ head
2. แบบ raw Ultralytics หนึ่งคลาส: `output0` ชนิด float รูปร่าง `[1,5,K]`
   (`center_x,center_y,width,height,confidence`) โดยขนาดต้องจับคู่กันดังนี้:
   `160 → K=525`, `320 → K=2100`, `640 → K=8400`

แบบ raw ต้องมี metadata `task=detect` และ `names` ที่ระบุ class `0` เพียงคลาสเดียว
ชื่อคลาส เช่น `Enemy` ใช้เพื่ออธิบายเท่านั้น ระบบจะ map เป็น player class `0`
และจะไม่สร้าง head class `7` เพิ่มเอง การคำนวณ NMS ใช้ NumPy ภายในโปรแกรม
ด้วย confidence ขั้นต่ำ `0.05`, IoU `0.45` และส่งออกไม่เกิน `300` กล่องต่อเฟรม
```

Also state in Thai that `[1,K,5]`, multi-class raw output, dynamic/rectangular tensors, arbitrary candidate counts, and missing/malformed metadata are rejected; external paths and models are never saved, copied, downloaded, or packaged.

Update the compile command by inserting `ai_yolo.py` immediately after `ai_detection.py`:

```powershell
python -m py_compile main.py ui.py motion.py combined_motion.py ai_targeting.py ai_tracking.py ai_detection.py ai_yolo.py ai_capture.py ai_zoom.py image_resize.py ai_service.py ai_model_selection.py display_timing.py overlay.py makcu_service.py hotkeys.py settings.py sound_service.py liquid_widgets.py distribution_metadata.py
```

- [ ] **Step 4: Synchronize repository guidance**

In `AGENTS.md`:

- Change the `ai_detection.py` layout description from a single fixed output contract to the exact dual-contract ONNX Runtime boundary.
- Add `ai_yolo.py` as the pure NumPy single-class raw decoder.
- Extend the external model rule to permit exact `[1,5,K]` output with the three size/count pairs and safe one-class detect metadata.
- State that legacy `[1,300,6]` behavior, downstream canonical `Detection`, and the bundled startup model remain unchanged.
- Add `ai_yolo.py` to the required `py_compile` command.

Do not weaken any prohibition on training, downloads, copied models, persistence, alternate runtimes, or package data.

- [ ] **Step 5: Run documentation, packaging, and source-policy tests**

Run:

```powershell
python -m unittest tests.test_entrypoints tests.test_distribution_metadata -v
python .\distribution_metadata.py --review-json
git diff --check
```

Expected: all tests PASS; review JSON includes `ai_yolo.py` in `compile_targets`; `nuitka_data_options` still contains only the bundled `models/all_games_320.onnx` file plus the pre-existing license/sound directory options; `git diff --check` reports no errors.

- [ ] **Step 6: Commit repository guidance and user documentation**

```powershell
git add README.md AGENTS.md tests/test_entrypoints.py
git commit -m "docs: explain raw one-class model support"
```

Confirm no `.onnx` path is staged with `git diff --cached --name-only`.

---

## Final Verification Gate

- [ ] **Step 1: Compile every reviewed source module**

Run:

```powershell
python -m py_compile main.py ui.py motion.py combined_motion.py ai_targeting.py ai_tracking.py ai_detection.py ai_yolo.py ai_capture.py ai_zoom.py image_resize.py ai_service.py ai_model_selection.py display_timing.py overlay.py makcu_service.py hotkeys.py settings.py sound_service.py liquid_widgets.py distribution_metadata.py
```

Expected: exit code `0` with no output.

- [ ] **Step 2: Run the complete hardware-free suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS with no failures or errors.

- [ ] **Step 3: Verify pinned runtime imports**

Run: `python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"`

Expected: exit code `0` with no import exception.

- [ ] **Step 4: Preserve the bundled release self-check**

Run: `python .\main.py --ai-runtime-self-check`

Expected: JSON reports `"ok": true`, the approved bundled SHA-256, and `"provider": "DmlExecutionProvider"`; this command must still inspect `models/all_games_320.onnx`, not the Apex model.

- [ ] **Step 5: Review the canonical distribution plan without building**

Run: `python .\distribution_metadata.py --review-json`

Expected: exit code `0`; compile targets include `ai_yolo.py`; model package data remains exactly `models/all_games_320.onnx=models/all_games_320.onnx`; do not run `gen.bat` or Nuitka.

- [ ] **Step 6: Validate the user-owned Apex model read-only through the production detector**

First verify the exact path remains present and untracked:

```powershell
Test-Path -LiteralPath 'C:\Users\User\Desktop\Jitter\models\Apex_20k_pictures_640.onnx'
git status --short -- 'models/Apex_20k_pictures_640.onnx'
```

Expected: `True` and an untracked `??` entry.

Then run one zero-frame production inference:

```powershell
python -c "import numpy as np; from ai_detection import OnnxDetector; p=r'C:\Users\User\Desktop\Jitter\models\Apex_20k_pictures_640.onnx'; d=OnnxDetector(p); result=d.detect(np.zeros((320,320,3), dtype=np.uint8)); print({'input_size': d.input_size, 'provider': d.provider, 'detections': len(result), 'classes': sorted({x.class_id for x in result})})"
```

Expected: `input_size` is `640`, `provider` is `DmlExecutionProvider`, inference returns without a contract error, `detections` is between `0` and `300`, and `classes` is either empty or `[0]` for a zero frame. No file is written, copied, or staged.

- [ ] **Step 7: Check the final diff, commits, and protected model boundary**

Run:

```powershell
git diff --check
git status --short
git log --oneline -5
git diff 7b5cf00..HEAD --stat
git diff 7b5cf00..HEAD -- models
```

Expected: source/tests/docs are committed; the user-owned `.onnx` files remain untracked and absent from every diff; no generated artifacts or external model paths were added; the three feature commits are visible after the design/plan commits.

Hardware-dependent Makcu checks remain explicitly unverified until a device is connected. Do not claim packaged-binary success because this plan does not authorize a Nuitka build.
