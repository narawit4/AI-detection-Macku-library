# Raw Single-Class Ultralytics YOLO Support Design

**Date:** 2026-08-29

**Status:** Approved in chat; awaiting written-spec review

## Summary

Jitter will continue to use the approved bundled post-NMS model at startup and
will additionally accept runtime-browsed, single-class Ultralytics detection
models whose raw output has the exact `[1,5,K]` layout used by
`Apex_20k_pictures_640.onnx`. A pure NumPy decoder will convert center-format
boxes, perform deterministic class-agnostic non-maximum suppression, map the
single model class to Jitter player class `0`, and publish no more than 300
canonical 320-space detections.

This is an additional external-model contract. It does not replace or weaken
the existing `[1,300,6]` post-NMS contract, change the bundled startup model,
add a runtime dependency, or package an external model.

## Goals

- Accept the user-owned `Apex_20k_pictures_640.onnx` at runtime.
- Support the same raw one-class Ultralytics layout at input sizes 160, 320,
  and 640 when every other contract field is valid.
- Decode raw `cx, cy, width, height, confidence` predictions with NumPy only.
- Apply bounded, deterministic NMS before publishing detections.
- Map the model's sole class, including the metadata label `Enemy`, to Jitter
  player class `0`.
- Preserve canonical 320-space targeting, Overlay, movement, and Adaptive Zoom.
- Preserve nearest-current-frame selection and detector-order tie-breaking.
- Preserve off-UI-thread validation, exact-ready switching, rollback-once,
  generation cancellation, and AI failure isolation.
- Preserve runtime-only model browsing with no external path or model format
  persistence.

## Non-Goals

- Do not add generic multi-class YOLO mapping.
- Do not infer player/head semantics from arbitrary class names.
- Do not synthesize head boxes from a player box.
- Do not add model conversion, export, training, downloads, or profiles.
- Do not add OpenCV, Pillow, Torch, Ultralytics, or another model runtime.
- Do not accept dynamic, rectangular, transposed, or arbitrary raw outputs.
- Do not add UI settings for NMS, model format, or class mapping.
- Do not copy, rename, persist, bundle, or package the Apex model.
- Do not change the bundled model hash or packaged DirectML self-check.

## Supported Detector Contracts

Every accepted model continues to have exactly one input and one output.

### Shared input contract

- Name: `images`
- Type: `tensor(float)`
- Shape: `[1,3,N,N]`
- `N`: exactly 160, 320, or 640
- Every dimension must be a static Python integer at the ONNX Runtime boundary.

### Existing post-NMS output contract

- Name: `output0`
- Type: `tensor(float)`
- Shape: `[1,300,6]`
- Rows: `x1, y1, x2, y2, confidence, class_id`
- Class `0`: player
- Class `7`: head

This path retains its current parser and behavior unchanged.

### New raw one-class output contract

- Name: `output0`
- Type: `tensor(float)`
- Shape: `[1,5,K]`
- Channels: `center_x, center_y, width, height, confidence`
- Candidate count is tied exactly to the static input size:
  - input 160: `K = 525`
  - input 320: `K = 2100`
  - input 640: `K = 8400`

The candidate counts are the fixed three-head Ultralytics layout for strides
8, 16, and 32. `[1,K,5]`, a different channel count, or a mismatched candidate
count is rejected rather than guessed.

## Ultralytics Metadata Contract

The raw path must not classify an arbitrary `[1,5,K]` tensor as YOLO solely by
shape. The session's custom model metadata must satisfy all of the following:

- `task` is exactly `detect`;
- `names` safely parses with `ast.literal_eval` to a dictionary;
- the dictionary contains exactly key `0`;
- the value for key `0` is a non-empty string.

The label itself is informational. `Enemy`, `enemy`, `person`, or another
single non-empty label maps to Jitter player class `0`; no name maps to head
class `7`. Metadata parsing never evaluates executable code.

The detector does not use the filename to select a format or class mapping.

## Module Boundaries

### `ai_yolo.py`

A new pure module owns raw one-class post-processing. It depends only on NumPy
and Jitter's immutable `Detection` value type. It has no Tkinter, ONNX Runtime,
DXCam, Makcu, filesystem, logging, or UI dependency.

Its narrow public behavior is equivalent to:

```python
decode_single_class_yolo(
    output: numpy.ndarray,
    input_size: int,
) -> tuple[Detection, ...]
```

Private helpers own stable confidence ordering and IoU suppression. Exact
constants are module-level and tested:

- minimum candidate confidence: `0.05`;
- NMS IoU threshold: `0.45`;
- maximum published detections: `300`.

### `ai_detection.py`

`OnnxDetector` remains the ONNX Runtime boundary. During construction it:

1. validates the shared input;
2. recognizes either the legacy post-NMS output or the raw one-class output;
3. validates Ultralytics metadata for the raw path;
4. stores a private immutable output-format decision for the generation.

During inference it preprocesses the same canonical frame, runs the same named
tensor, then calls the parser selected during construction. Runtime output is
still shape-checked; a session that changes shape after validation fails closed.

No UI metadata determines the parser. `AiService` constructs its own detector
for every generation and therefore revalidates the selected model.

### Existing downstream components

`ai_targeting.py`, `ai_zoom.py`, `ai_service.py`, `overlay.py`, and combined
motion continue to consume ordinary immutable `Detection` values in canonical
320-space. They do not branch on model format.

## Raw Decode and NMS Algorithm

The pure decoder performs the following steps in order:

1. Require an owned or borrowed numeric NumPy array with the exact raw shape
   for `input_size`.
2. View the tensor as `K` candidate rows without mutating the inference output.
3. Reject non-finite rows.
4. Retain confidence values in `[0.05, 1.0]`.
5. Require strictly positive width and height.
6. Convert center-format boxes to `x1, y1, x2, y2` in model-input space.
7. Stable-sort by descending confidence; equal confidence preserves raw
   candidate order.
8. Repeatedly keep the highest-ranked remaining box and suppress boxes whose
   IoU with it is strictly greater than `0.45`.
9. Stop as soon as 300 boxes have been kept or no candidates remain.
10. Sort the kept raw candidate indexes into original detector order.
11. Scale coordinates by `320 / input_size`, clip them to `[0,320]`, and reject
    boxes that collapse after clipping.
12. Publish each survivor as `Detection(..., class_id=0)`.

The implementation must use vectorized one-to-many IoU calculations per kept
box. It must not allocate a full `K x K` IoU matrix. The loop is bounded by
`MAX_DETECTIONS`, so a pathological high-confidence tensor performs at most 300
suppression iterations.

Filtering at `0.05` cannot remove a detection that the UI would accept because
`0.05` is the minimum validated Jitter confidence setting. The existing target
selection applies the user's live threshold after decoding and remains the
authority for movement and Overlay selection.

## Targeting Semantics

Every raw detection is a player box:

- target area `head`: use the existing player fallback aim point at 20 percent
  of box height below the top edge;
- target area `upper_body`: use the existing player upper-body aim point;
- target area `chest`: use the existing player chest aim point.

The raw model never creates a class-7 head detection. The Head Boxes Overlay
visibility option therefore does not hide its player boxes. Accepted boxes
still compete by minimum Euclidean distance to `(160,160)`, with the raw model's
restored candidate order as the exact-distance tie-break.

## Adaptive Zoom

Adaptive Zoom remains canonical and format-independent:

- base capture is 320 by 320;
- the detector internally resizes the base frame to its static input;
- raw output is decoded back to canonical 320-space;
- zoom crop geometry consumes the selected canonical player box;
- the canonical refinement frame is passed through the same raw detector;
- refinement output is decoded and associated with the selected base box using
  the existing same-frame rules.

The model's lack of head class does not change 1.0x/1.5x/2.0x eligibility for
player targets. Refinement misses, cooldown, stability confirmation, and base
fallback remain unchanged.

## Model Selection and UI

No new control or persisted field is added. Browse validation automatically
recognizes either supported output contract. The model label continues to show
only Default/Custom, filename, and validated input size; it does not claim an
output format.

Validation and runtime errors use concise authored messages that identify the
two accepted output families without exposing a full external path. Full model
path, metadata, provider, and exception diagnostics remain in `app.log`.

The existing switch lifecycle remains authoritative:

1. browse creates an untrusted runtime-only choice;
2. AI demand pauses for validation;
3. validator constructs the detector off the UI thread;
4. the exact token/path ready event may start a candidate generation;
5. candidate readiness commits the choice;
6. candidate failure triggers one rollback attempt;
7. terminal rollback failure follows existing AI isolation behavior.

STOP, disconnect, source changes, Master disable, Test 3s, and shutdown retain
their cancellation barriers.

## Persistence and Packaging

There is no configuration schema change. Jitter does not persist external model
path, class label, output format, input size, NMS state, or decoded targets.

`Apex_20k_pictures_640.onnx` and every other external model remain user-owned,
untracked runtime inputs. The canonical Nuitka command continues to include
only:

```text
--include-data-files=models/all_games_320.onnx=models/all_games_320.onnx
```

The new `ai_yolo.py` source module is added to compile/review inventories, but
no raw model is added to package data.

The source and packaged `--ai-runtime-self-check` continue to require the exact
approved bundled SHA-256, the legacy `[1,300,6]` output contract, and
`DmlExecutionProvider`. Raw external support must not relax release validation.

## Error Handling

Raw models fail validation for any of the following:

- missing or malformed `task`/`names` metadata;
- more than one class or a class key other than integer `0`;
- wrong tensor name or type;
- wrong orientation, channel count, candidate count, or static dimension type;
- dynamic or unsupported input size;
- malformed runtime output;
- nonnumeric output arrays.

Malformed individual prediction rows are skipped when the tensor-level
contract is otherwise valid. If every row is invalid or below the fixed floor,
the decoder publishes an empty tuple for the current frame; it does not hold an
old target.

An in-service inference or decoder exception follows the existing AI runtime
failure path: clear AI state, hide Overlay, deselect AI Aim, retain eligible
Jitter, or disarm Master for AI-only operation.

## Test Strategy

Implementation follows test-driven development.

### Pure decoder tests

- accept exact 160/320/640 raw shapes;
- reject wrong candidate counts, orientation, dtype behavior, and malformed
  tensors;
- decode hand-derived center boxes;
- filter non-finite, low-confidence, out-of-range-confidence, and non-positive
  boxes;
- scale and clip at every supported input size;
- suppress overlapping boxes at the exact IoU boundary;
- preserve equal-confidence and final detector-order ties;
- stop at 300 survivors;
- prove no full pairwise IoU matrix is required by the helper interface;
- publish only class `0` detections.

### Detector contract tests

- retain every existing post-NMS contract test;
- recognize the exact raw shape for each supported input;
- reject raw shape/input-size mismatches;
- require safe one-class detect metadata;
- reject executable or malformed metadata without evaluating it;
- select the correct parser for each validated session;
- re-check runtime output shape;
- preserve DML-first and CPU fallback behavior.

### Integration tests

- nearest-current-frame targeting receives raw player detections immediately;
- Head target area uses existing player fallback;
- Overlay and head-box visibility retain class semantics;
- Adaptive Zoom base and refinement remain canonical;
- stale generations cannot publish decoded results;
- Jitter continues when raw AI has no accepted target;
- validation, candidate startup, rollback, STOP, and Test 3s behavior remain
  unchanged.

### Real-model verification

When the user-owned Apex model is present locally, run a read-only validation
and one zero-frame inference through `OnnxDetector`. Confirm input size 640,
DirectML readiness, successful raw decoding, and no persistence or staging of
the model. This is local verification, not a repository test fixture.

### Repository verification

Run the complete compile, unit-test, dependency-import, source self-check,
distribution review, and diff checks required by `AGENTS.md`. Connected Makcu
verification remains required before claiming hardware behavior or producing a
release.

## Compatibility and Rollout

Existing bundled and external `[1,300,6]` models remain unchanged. Exact raw
single-class Ultralytics models become an additional accepted family. Generic
multi-class raw YOLO, alternate output orientation, and arbitrary export formats
remain explicitly unsupported.

The change requires no configuration migration. Every launch still starts with
the bundled 320 post-NMS model, and Use Default remains the stable rollback path.
