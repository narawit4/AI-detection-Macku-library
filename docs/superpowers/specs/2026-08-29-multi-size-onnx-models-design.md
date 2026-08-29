# Multi-Size External ONNX Model Support Design

**Date:** 2026-08-29

**Status:** Approved for implementation planning

## Summary

Jitter will continue to start with the approved bundled 320-by-320 model while
allowing `Browse...` to select runtime-only external ONNX models whose static
square input is exactly 160, 320, or 640 pixels. Capture, Overlay, targeting,
movement, and Adaptive Zoom retain a canonical 320-by-320 coordinate space.
The detector alone adapts input images to the selected model size and maps
model output boxes back into canonical coordinates before any downstream
consumer sees them.

This design preserves the current field of view, crosshair at `(160, 160)`,
response curve, movement feel, Overlay alignment, and zoom geometry while
supporting the three requested model contracts.

## Goals

- Accept external ONNX models with input shapes `[1,3,160,160]`,
  `[1,3,320,320]`, or `[1,3,640,640]`.
- Keep the bundled `models/all_games_320.onnx` model as the startup default.
- Detect an external model's supported input size automatically during the
  existing off-UI-thread validation flow.
- Resize the canonical 320-by-320 RGB frame to the selected model input size
  using NumPy only.
- Map all detection box coordinates back to canonical 320-by-320 coordinates.
- Keep nearest-current-frame target selection unchanged after mapping.
- Keep Overlay, response-curve, movement-servo, and Adaptive Zoom geometry
  unchanged.
- Show the validated model size in the runtime model label.
- Preserve generation safety, exact-ready model switching, rollback-once,
  immediate cancellation, and safe runtime-error behavior.
- Preserve runtime-only external model selection with no path persistence,
  copying, packaging, or download behavior.

## Non-Goals

- Do not bundle or package additional 160-by-160 or 640-by-640 models.
- Do not add model downloads, model training, profiles, or model-path
  persistence.
- Do not accept 128, 256, dynamic, rectangular, or arbitrary input sizes.
- Do not change the physical centered capture region from 320 by 320 pixels.
- Do not resize the Overlay window or change its on-screen alignment.
- Do not give a 640 model a wider field of view or additional source pixels;
  its input is an upscale of the canonical 320 frame.
- Do not alter the approved bundled model hash or packaged self-check contract.
- Do not add OpenCV, Pillow, Torch, Ultralytics, or another image/model runtime.
- Do not persist model size as configuration; it is derived runtime metadata.

## Accepted Model Contract

Every accepted model has exactly one input and one output.

### Input

- Name: `images`
- Type: `tensor(float)`
- Rank and layout: `[1, 3, N, N]`
- Static size: `N` must be exactly one of `160`, `320`, or `640`

Dynamic dimensions, symbolic dimensions, non-square dimensions, alternate
batch sizes, alternate channel counts, and alternate tensor types are rejected.

### Output

- Name: `output0`
- Type: `tensor(float)`
- Exact shape: `[1, 300, 6]`
- Class `0`: player
- Class `7`: head

The output contract remains fixed for all three supported input sizes.

## Canonical Coordinate Space

All application components outside the detector continue to use a logical
320-by-320 coordinate space:

- The centered crosshair is `(160, 160)`.
- DXCam captures a centered 320-by-320 RGB frame.
- Target aim points are expressed in the 0-320 coordinate range.
- The response curve uses the existing 320-space reference radius.
- The movement servo subtracts the existing center value of 160.
- Overlay boxes remain aligned with the physical 320-by-320 capture region.
- Adaptive Zoom receives and publishes canonical 320-space frames and boxes.

No downstream component needs to branch on model input size.

## Shared RGB Resize Primitive

A new pure module, `image_resize.py`, will own the existing NumPy bilinear RGB
resize implementation currently located in `ai_zoom.py`.

It will expose a narrow function equivalent to:

```python
resize_rgb_bilinear(image: numpy.ndarray, output_size: int) -> numpy.ndarray
```

The helper will:

- accept non-empty `H x W x 3` NumPy arrays;
- require a positive integer output size;
- return a contiguous `uint8` square RGB image;
- preserve the existing deterministic round-to-nearest behavior;
- cache separable resize plans where useful;
- remain independent of Tkinter, Makcu, ONNX Runtime, and capture code.

`ai_zoom.py` and `ai_detection.py` will both import this helper. Moving the
existing implementation avoids maintaining two resize algorithms and avoids a
dependency from detection into zoom.

## Detector Behavior

`OnnxDetector` will inspect and validate the model session during construction.
After validation it exposes an immutable/read-only `input_size` value of 160,
320, or 640.

The public detection path continues to accept only a canonical RGB
`(320, 320, 3)` `uint8` frame. Before NCHW conversion:

- input size 160: resize 320 to 160;
- input size 320: use the canonical frame without a spatial resize;
- input size 640: resize 320 to 640.

The resulting image is converted to contiguous float32 NCHW and normalized to
the existing 0-1 range. The input/output tensor names and provider behavior are
unchanged.

### Output Mapping

Output box coordinates are treated as pixels in model-input space and mapped
to canonical 320 space using:

```text
scale = 320 / input_size
canonical_coordinate = model_coordinate * scale
```

Therefore:

- 160 output coordinates are multiplied by `2.0`;
- 320 output coordinates are unchanged;
- 640 output coordinates are multiplied by `0.5`.

Coordinates are mapped before canonical clipping and invalid-box rejection.
Confidence and class values are not scaled. Parsed `Detection` objects always
contain canonical coordinates.

The detector continues to reject malformed output arrays, non-numeric rows,
non-finite values, and boxes that remain empty after mapping and clipping.

## Model Choice and Validation Metadata

`ModelChoice` remains immutable and gains runtime input-size metadata:

- The bundled choice is known to be 320 at construction.
- A newly browsed external choice begins unvalidated with no trusted size.
- The background validator constructs `OnnxDetector`, reads its validated
  `input_size`, and publishes a validated choice containing that size.
- Only the exact validation event for the active switch token may update UI
  state.
- `AiService` constructs its own detector for the selected path and validates
  the contract again at runtime; it does not trust UI metadata for inference.

This keeps validation display metadata useful without weakening the service's
runtime boundary.

## UI Behavior

The existing model row and controls remain. No separate size selector is added.
Size is derived from the selected file.

Labels use concise runtime metadata:

```text
Default · all_games_320.onnx · 320×320
Custom · example.onnx · 160×160
Custom · example.onnx · 640×640
```

While an external candidate is still validating, the existing loading label is
used without presenting an untrusted size. A successful ready event commits the
validated choice and size. Footer errors remain concise and never expose a full
external path; detailed path and exception diagnostics remain in `app.log`.

## Model Switch Lifecycle

The existing lifecycle remains authoritative:

1. `Browse...` creates an external runtime choice without persisting it.
2. Jitter pauses eligible AI motion/runtime for the model switch.
3. The background validator checks the exact multi-size contract.
4. A valid candidate starts a fresh `AiService` generation.
5. The exact ready event commits the candidate and restarts eligible motion.
6. Candidate startup failure triggers exactly one rollback attempt to the
   previous model.
7. Terminal rollback failure follows the existing safe AI runtime-error path.

STOP, disconnect, source changes, Master disable, Test 3s, and shutdown retain
their current generation and cancellation barriers.

## Adaptive Zoom Interaction

Adaptive Zoom remains entirely canonical:

- Base capture is 320 by 320.
- Base inference receives a detector that performs any model resize internally.
- Parsed base detections return in canonical coordinates.
- Zoom crop geometry uses canonical 320-space targets.
- The zoom crop is resized to a canonical 320-by-320 refinement frame by the
  shared resize helper.
- The detector then adapts that refinement frame to 160, 320, or 640.
- Refined output is mapped back to canonical space before same-frame
  association and composition.

The current rules for 1.0x/1.5x/2.0x, stability confirmation, 100 ms cooldown,
refinement fallback, selected-base association, and unrelated Overlay boxes do
not change.

## Capture, Overlay, Targeting, and Movement

The following behaviors remain unchanged:

- DXCam capture region and frame validation stay fixed at 320 by 320.
- Overlay stays centered, click-through, capture-excluded, and 320 by 320.
- All accepted head and player aim points compete together each base frame.
- Selection uses minimum Euclidean distance to `(160, 160)` with detector order
  as the exact-distance tie break.
- There is no previous-frame identity, ambiguity hold, recovery confirmation,
  or replacement delay for movement publication.
- Response Curve, Strength, Smoothing, Max Step, acceleration, fractional
  accumulation, freshness, and combined Jitter composition remain unchanged.

## Cadence and Performance Status

Display-derived capture and servo cadence policy does not change. The measured
inference FPS already reflects the selected model's real throughput and remains
runtime-only status.

The README will explain expected trade-offs without guaranteeing performance:

- 160 generally requires less inference work and may run faster;
- 320 remains the balanced bundled default;
- 640 generally requires more inference work and may run slower;
- actual latency and detection quality depend on the model and hardware;
- all three sizes see the same physical 320-by-320 source region.

## Persistence and Packaging

No configuration schema change is required.

The application must not persist:

- external model path;
- external model input size;
- selected model state;
- model validation metadata;
- provider, FPS, cadence, target, snapshot, or zoom state.

Every launch starts with the bundled 320 model. External 160/320/640 models are
never copied into `models/`, downloaded, bundled, or packaged.

The source and packaged `--ai-runtime-self-check` continue to require the exact
approved bundled model hash, `[1,3,320,320]` input, `[1,300,6]` output, and
`DmlExecutionProvider`. Multi-size external support must not relax the release
self-check.

The new `image_resize.py` source module must be included in compile/review and
packaging source inventories. The canonical Nuitka plan includes only
`models/all_games_320.onnx` through an exact data-file option; it never packages
the `models/` directory recursively.

## Error Handling

External candidates are rejected safely for:

- missing file or wrong extension;
- model-session construction failure;
- wrong input/output count;
- wrong input/output name or type;
- input size outside 160/320/640;
- dynamic, rectangular, or otherwise malformed input shape;
- output shape other than `[1,300,6]`;
- candidate startup failure after validation.

UI messages remain short and actionable, for example that the selected model
must use a 160, 320, or 640 square input. Full diagnostics, including the
external path and underlying exception, go only to `app.log`.

An in-service inference/runtime failure retains the existing behavior: clear AI
snapshots, hide Overlay, deselect AI Aim, continue/restart selected Jitter when
eligible, and disarm Master for an AI-only failure.

## Test Strategy

Implementation follows test-driven development.

### Pure Resize Tests

- Preserve current deterministic bilinear output for zoom's existing resize
  cases after moving the helper.
- Produce contiguous uint8 RGB output at 160, 320, and 640.
- Reject malformed input and invalid output sizes.

### Detection Tests

- Accept exact 160, 320, and 640 input contracts.
- Reject 128, 256, dynamic, rectangular, wrong-layout, wrong-type, and
  wrong-name inputs.
- Produce normalized float32 NCHW tensors with shapes matching each model.
- Verify 160-to-320 coordinate scaling by 2.0.
- Verify 320 coordinates remain unchanged.
- Verify 640-to-320 coordinate scaling by 0.5.
- Verify canonical clipping and empty-box rejection after scaling.
- Verify `OnnxDetector.input_size` exposes only validated metadata.

### Model Selection and UI Tests

- Bundled choice reports size 320.
- External choice does not claim a size before validation.
- Validator publishes the exact detected supported size.
- Stale/cancelled validation cannot publish size or choice state.
- Model labels show 160×160, 320×320, or 640×640 only after validation.
- Invalid-size candidates keep the previous model without leaking paths.
- Switch, exact-ready commit, rollback-once, Test 3s exclusion, and button
  availability remain correct.
- No model path or size enters configuration saves.

### Service and Zoom Tests

- Service publication remains canonical for fake 160/320/640 detectors.
- Nearest-current-frame selection receives mapped boxes and remains unchanged.
- Adaptive Zoom base/refinement composition remains canonical at all sizes.
- Crowded refinement stays associated with the already-selected base target.
- Generation cancellation prevents obsolete resized inference from publishing.
- Combined Jitter continues when an AI frame has no valid target.

### Distribution and Documentation Tests

- Add `image_resize.py` to canonical compile/review inventories.
- Preserve bundled model and DirectML self-check assertions.
- Ensure no alternate models or paths enter package data.
- Update README and AGENTS.md with the accepted sizes and fixed canonical
  capture behavior.

## Verification

Run the repository's complete verification commands:

```powershell
python -m py_compile main.py ui.py motion.py combined_motion.py image_resize.py ai_targeting.py ai_tracking.py ai_detection.py ai_capture.py ai_zoom.py ai_service.py ai_model_selection.py display_timing.py overlay.py makcu_service.py hotkeys.py settings.py sound_service.py liquid_widgets.py distribution_metadata.py
python -m unittest discover -s tests -v
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
git diff --check
```

Hardware verification with a connected Makcu additionally covers one external
model of each supported input size:

- validation and displayed size;
- DirectML/CPU provider readiness as applicable;
- Trigger and Modifier gating;
- nearest-current-frame head/player selection;
- Overlay alignment and head-box visibility;
- 1.0x/1.5x/2.0x zoom behavior;
- Jitter-only, AI-only, and combined movement;
- STOP, disconnect, model switch, rollback, Test 3s, and shutdown;
- return to the bundled default model.

## Compatibility and Rollout

The change requires no config migration. Existing schema 1-5 behavior and
future-schema protection remain unchanged. The existing bundled default and
its exact runtime self-check remain the stable rollback path.

Existing external 320 models that match the current contract remain valid.
Previously rejected matching 160 and 640 models become valid. Models using 128,
256, dynamic, or arbitrary sizes remain explicitly unsupported.
