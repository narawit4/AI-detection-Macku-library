# Full Primary-Display Detection Design

**Date:** 2026-08-31

**Status:** Approved for implementation

## Summary

Jitter will replace its physical centered 320-by-320 capture with one native-
resolution capture of the complete primary display. Each captured frame will
be letterboxed once into the active model's existing square 160, 320, or 640
input without stretching. Both supported detector output contracts will be
mapped through the inverse letterbox transform into source-screen coordinates
before targeting, movement, Adaptive Zoom composition, or Overlay rendering.

The base path remains one inference per captured frame. There is no tiled
scanning. Adaptive Zoom may still perform its existing single same-frame
refinement inference when its movement gate is eligible. The fixed 1,000 Hz
motion servo remains independent of capture and inference cadence.

This is an explicit replacement for the earlier centered-capture design. It
does not change model selection, ONNX contracts, source gating, persistence,
packaging, or the runtime safety lifecycle.

## Approved Decisions

- Detect the complete primary display, not all monitors.
- Use one full-display base inference per captured frame.
- Preserve display aspect ratio with letterboxing; never stretch the frame.
- Do not add tiled or rotating-region inference.
- Do not add a capture-area control or persist capture geometry.
- Keep the bundled 320 model as the startup default and preserve runtime-only
  browsing of validated 160, 320, and 640 external models.
- Keep Adaptive Zoom, but allow at most its existing one extra refinement
  inference on an eligible frame.
- Keep the fixed 1,000 Hz motion servo and current immediate-cancellation
  behavior.

## Goals

- Give base detection the complete field of view of the primary display.
- Keep boxes aligned with the full-screen Overlay at every supported display
  aspect ratio.
- Preserve detector order as the exact-distance target-selection tie break.
- Normalize the response curve against the actual frame center-to-corner
  radius.
- Make full-screen and zoom-refinement coordinate transforms pure,
  deterministic, and hardware-free testable.
- Keep one source of truth for frame geometry in every published detection and
  target snapshot.
- Preserve generation safety, target freshness, source composition, and all
  STOP/disconnect/shutdown barriers.

## Non-Goals

- Multi-monitor or virtual-desktop capture.
- Tiled scanning, overlapping tiles, or queued scan regions.
- Recovering an object that the full-field base inference did not detect.
- Changing the accepted ONNX input/output contracts or class semantics.
- Adding a model download, training, profile, persistence, or packaging path.
- Adding OpenCV, Pillow, Torch, Ultralytics, or another resize/model runtime.
- Changing capture cadence into motion cadence or sending zero Makcu deltas.
- Persisting display dimensions, letterbox transforms, detections, targets,
  cadence, provider, or zoom state.

## Coordinate Spaces

The implementation will use explicit coordinate spaces instead of treating
all coordinates as an implicit 320-by-320 square.

### Source-screen space

Source-screen space is the native RGB frame returned for output index 0:

```text
width  = captured frame width
height = captured frame height
center = (width / 2, height / 2)
```

Published `Detection` boxes and `TargetSnapshot` aim points use this space.
Their valid bounds are `0..width` and `0..height`. The dimensions travel with
the immutable frame and target snapshots so a consumer never combines a box
with stale display geometry.

### Model-input space

Model-input space remains a square of side `N`, where `N` is exactly 160, 320,
or 640. A pure immutable letterbox transform records:

- source width and height;
- model input size;
- resized content width and height;
- left/top padding and the complementary right/bottom padding;
- exact horizontal and vertical inverse scales implied by the integer resize.

The transform is calculated independently for every frame shape. Repeated
shapes may use bounded cached resize plans, but cached state is never treated
as runtime geometry authority.

### Logical zoom-policy space

The existing Adaptive Zoom thresholds were defined in a 320-square view.
Geometry-sensitive zoom decisions will therefore map source points and box
sizes into a temporary aspect-preserving 320-square letterbox space. This
keeps the 96-pixel center-distance rule, size thresholds, 18-pixel stability
rule, and fixed association margins resolution-independent without exposing
logical coordinates to the Overlay or movement engine.

## Full-Display Capture

`DxcamCapture` will create output index 0 with the existing RGB NumPy backend
and short latest-frame buffer. It will start capture over the complete output:

```text
(0, 0, camera.width, camera.height)
```

Each returned frame must be a non-empty `H x W x 3` `uint8` RGB array. The
capture wrapper returns an owned array and does not retain frame views. It no
longer requires `(320, 320, 3)`.

The target FPS remains the display-derived runtime policy capped at 240 FPS.
The worker continues to consume only the latest available frame and never
queues catch-up capture or inference work.

If DXCam cannot start full-output capture, or supplies malformed geometry or
pixels, the existing AI runtime-error path remains authoritative. No movement
may be produced from malformed frame data.

## Aspect-Preserving Letterbox

For source dimensions `W x H` and model side `N`, the pure preprocessing path
will:

1. compute `gain = min(N / W, N / H)`;
2. round the resized width and height deterministically to positive integers
   no larger than `N`;
3. split unused pixels deterministically across opposite sides, with the extra
   pixel placed on the right or bottom when padding is odd;
4. resize RGB content with the shared deterministic NumPy bilinear primitive;
5. place it in an `N x N x 3` RGB canvas filled with value 114;
6. convert the result to contiguous normalized float32 NCHW.

For example, a 1920-by-1080 frame becomes 320 by 180 content with 70 pixels of
padding above and below for the bundled 320 model. A 1080-by-1920 portrait
frame becomes 180 by 320 content with 70 pixels of padding on each side.

The resize module will support an explicit rectangular output size while
retaining its current square convenience behavior and deterministic rounding.
Resize work is proportional to the model-sized output, not to a model-sized
copy for every source pixel.

## Detector Output Mapping

`OnnxDetector.detect(frame)` will accept any valid non-empty RGB `uint8` frame
and return detections in that frame's source coordinate space.

Both output formats first produce boxes in model-input coordinates:

- legacy post-NMS `[1,300,6]`;
- raw single-class `[1,5,K]` after the existing confidence handling and NMS.

A single shared inverse-letterbox stage will then:

1. intersect each model-space box with the resized-content rectangle;
2. discard a box whose intersection is empty, including padding-only boxes;
3. subtract left/top padding;
4. divide x coordinates by the exact resized-width/source-width ratio and y
   coordinates by the exact resized-height/source-height ratio;
5. clamp the result to source bounds;
6. reject any box that is empty after mapping.

Confidence and class values are unchanged. Mapping happens before downstream
confidence filtering and target selection. Legacy and raw models therefore
share exactly the same source-space boundary.

Model validation, DirectML-first provider selection, raw metadata parsing,
candidate rollback, and startup-default model behavior remain unchanged.

## Target Selection and Snapshots

`analyze_detections` will receive the current frame dimensions explicitly. It
will preserve the current behavior of:

- filtering by confidence and supported class;
- deriving the configured aim point for every accepted head and player;
- considering heads and players together;
- choosing minimum Euclidean distance for the current frame only;
- preserving detector order for an exact-distance tie;
- publishing no history hold, recovery delay, or identity preference.

The comparison center changes from fixed `(160, 160)` to
`(frame_width / 2, frame_height / 2)`.

`DetectionFrameSnapshot` and `TargetSnapshot` will carry validated positive
frame width and height. Compatibility construction used by existing pure tests
may default to 320 by 320, but production publication always supplies the
captured dimensions.

## Movement Response

The 1,000 Hz `AimMovementEngine` will consume source-space target coordinates.
For each fresh target it computes:

```text
error_x         = aim_x - frame_width / 2
error_y         = aim_y - frame_height / 2
reference_radius = hypot(frame_width / 2, frame_height / 2)
normalized       = min(1, hypot(error_x, error_y) / reference_radius)
```

The existing five-point curve, Strength, time-based Smoothing, Max Step,
acceleration bound, fractional carry, 150 ms freshness, clamping, and excess-
discard rules then apply. This makes every screen corner the 100% curve point
while retaining the true source-pixel direction vector.

Invalid dimensions cause an immediate engine reset and zero output. A frame-
dimension change also resets prior velocity, fractional carry, and remaining
error before the new target is consumed. It must never mix motion state from
two display geometries.

The servo schedule itself is unchanged: absolute 1 ms deadlines, missed-slot
skipping, no catch-up movement, no zero-delta Makcu call, and immediate stop
signalling.

## Adaptive Zoom

Full-field acquisition always runs first. Adaptive Zoom cannot discover a
target absent from that base result.

When the existing connected/Master/AI-selected/Trigger/Modifier gate is true,
the selected base target may request one 1.5x or 2.0x refinement. Zoom-factor,
stability, and compatibility decisions use the temporary logical 320-square
policy mapping described above.

The refinement crop will:

- preserve the source frame's aspect ratio;
- use approximately `W / factor` by `H / factor` source pixels;
- center on the selected aim point and clamp to source bounds;
- remain an owned contiguous RGB crop;
- be letterboxed by the detector into the active model input exactly like a
  base frame.

The detector returns refined boxes in crop-local pixels. A rectangular
`ZoomTransform` records the crop origin and dimensions; composition maps a
compatible refined box back to full source-screen space by adding that origin.
Unrelated base detections remain unchanged.

Recoil confirmation, 1.5x limiting, 100 ms cooldown, normal-miss reset,
same-frame base fallback, generation-local stability, and `ZOOM` runtime status
remain unchanged. Overlay-only inference, idle operation, and `Test 3s` never
request the second inference.

## Overlay Mapping

The Overlay remains a primary-display-sized, click-through,
capture-excluded window. It no longer adds a centered 320-pixel capture offset.

For each fresh snapshot it scales source coordinates to the current canvas:

```text
canvas_x = source_x * canvas_width  / snapshot.frame_width
canvas_y = source_y * canvas_height / snapshot.frame_height
```

This is normally a 1:1 mapping and also prevents stale alignment if Tk reports
a display size change between capture and rendering. Every drawn coordinate is
clamped to the canvas. Head/player visibility, labels, widths, colors, selected
box emphasis, HUD position, HUD metric filters, and the 150 ms stale-frame
`NONE` rule are unchanged.

## Cadence and Expected Accuracy

The base path performs exactly one inference for each frame it processes. It
does not tile, overlap, or retain unprocessed frames. Eligible Adaptive Zoom
may raise that frame's total to two calls, exactly as before.

The 320 startup model sees the complete display compressed into its square
input. On a widescreen display its content occupies fewer than 320 vertical
model pixels, so very small or distant targets can be harder to detect than in
the old centered crop. This is an accepted tradeoff of full coverage with one
inference. A runtime-browsed compatible 640 model can retain more spatial
detail, but Jitter will not select, copy, persist, download, or package one
automatically.

Published base-frame cadence (reported as FPS), display cadence, provider, and
zoom factor remain runtime status only.

## UI and Configuration

No dashboard or Overlay-customizer control is added. Full-primary-display
detection is the sole capture mode.

There is no configuration-schema change. No display size, capture mode,
letterbox state, target, box, model path, or runtime cadence is serialized.
Every launch still begins with no selected source, Overlay off, and the bundled
320 model.

All existing Trigger, Modifier, Master, hotkey, STOP, Test 3s, source-selection,
model-switch, and shutdown semantics remain unchanged.

## Error and Cancellation Behavior

- STOP immediately cancels movement, hides Overlay, and removes its inference
  demand.
- Disable, disconnect, source changes, model switching, and shutdown retain
  their existing generation barriers and immediate cancellation.
- A malformed full-screen frame or unsafe geometry enters the existing AI
  runtime-error path rather than publishing partial coordinates.
- An AI runtime error hides Overlay and deselects AI Aim; eligible Jitter may
  continue through the current shared gate.
- Stale results from obsolete AI generations remain ignored.
- A failed refinement preserves the same-frame full-field base result.

## Planned Source Impact

- `jitter_app/ai/capture.py`: replace centered-region capture and fixed-shape
  validation with owned full-output RGB frames.
- `jitter_app/ai/resize.py`: add deterministic rectangular resize support.
- `jitter_app/ai/detection.py`: create letterbox preprocessing and shared
  inverse mapping for both output contracts.
- `jitter_app/ai/yolo.py`: expose/model-space raw decode before shared mapping.
- `jitter_app/ai/targeting.py`: carry frame geometry, select around the real
  center, and use a dynamic response radius.
- `jitter_app/ai/zoom.py`: use logical policy geometry and rectangular native
  crops/transforms.
- `jitter_app/ai/service.py`: pass per-frame geometry through base analysis and
  zoom publication.
- `jitter_app/presentation/overlay.py`: render source-space boxes across the
  full primary-display canvas.
- `README.md` and `AGENTS.md`: document full-primary-display behavior and
  remove runtime claims that capture remains physically centered 320 by 320.
- Existing capture, resize, detection, YOLO, targeting, movement, zoom,
  service, Overlay, UI, and distribution tests: update or extend the geometry
  contract without weakening unrelated assertions.

No packaging command or dependency change is required.

## Test Strategy

Implementation will follow failing-test-first slices.

### Pure geometry and resize

- exact 1920x1080, 2560x1440, ultrawide, square, and portrait letterbox plans;
- deterministic odd-padding placement and one-pixel minimum content;
- rectangular bilinear output shape, dtype, ownership, contiguity, and
  deterministic rounding;
- source-to-model-to-source point and box round trips within the expected
  integer-resize tolerance;
- padding-only rejection and content-edge clipping.

### Capture and detector

- DXCam receives the complete primary-output region and existing cadence;
- valid native frames are owned and malformed frames are rejected;
- 160/320/640 preprocessing preserves aspect ratio;
- legacy and raw output boxes share inverse-letterbox mapping;
- invalid outputs and model contracts remain rejected;
- DirectML-first construction and CPU fallback behavior remain unchanged.

### Targeting and movement

- nearest target is measured from the real center at multiple resolutions;
- exact-distance ties preserve detector order;
- screen corners evaluate at 100% response distance;
- 320x320 compatibility behavior remains exact;
- dimension changes reset velocity and fractional carry;
- freshness, dead zone, acceleration, Max Step, and direction-safe carry remain
  bounded at the fixed servo rate.

### Adaptive Zoom and service

- 1.5x/2.0x crops preserve source aspect ratio and clamp at every edge;
- crop-local detections compose back into full-screen source coordinates;
- unrelated base boxes remain unchanged;
- base result survives refinement miss or cancellation;
- one base call occurs per processed frame and no more than one eligible
  refinement call is added;
- Overlay-only, idle, and Test 3s exclude refinement;
- generation changes discard obsolete geometry and results.

### Overlay and integration

- full-screen boxes align at 1:1 dimensions and scale safely during a reported
  display-size change;
- all display corners and clipped boxes remain on-canvas;
- HUD, labels, filters, selected highlighting, visibility controls, and stale
  lock status remain correct;
- STOP, runtime error, source changes, disconnect, and shutdown retain their
  immediate behavior.

After implementation, run the repository's complete canonical verification
commands from `AGENTS.md`. Nuitka packaging and connected Makcu acceptance are
not part of ordinary implementation verification unless separately requested.

## Acceptance Criteria

- The base detector receives the entire primary display through one
  aspect-preserving letterboxed inference.
- A known source-screen box maps through every supported model size and renders
  at the correct full-screen Overlay location.
- Target selection uses the actual center of the captured frame.
- Movement response reaches the curve's 100% point at every source-screen
  corner without changing the fixed 1,000 Hz scheduling policy.
- Adaptive Zoom performs at most one additional eligible inference and maps its
  selected replacement box back into the full-screen frame.
- No tiling, model-contract expansion, dependency, config-schema, persistence,
  or packaging change is introduced.
- The canonical compile, full test suite, runtime imports, DirectML self-check,
  and distribution review all pass.
