# Adaptive AI Zoom Design

Date: 2026-08-27

## Context

Jitter currently captures the centered 320-by-320 primary-screen region and
runs the fixed bundled model once for each published AI frame. Real-screen
probing showed that enlarging a central crop can raise the best confidence for
small heads, but a permanently zoomed input loses detections outside the
narrower field of view. A display-only Overlay zoom cannot improve inference
because it does not change the pixels presented to the model.

Adaptive AI Zoom therefore keeps the full 320-by-320 pass for acquisition and
adds a second, target-centered inference pass only while the normal movement
gate is active and the selected object is small. The refined result is mapped
back into the original capture coordinates. The fixed model, fixed capture
contract, DirectML provider policy, and approved dependencies remain unchanged.

## Goals

- Improve localization of small distant heads and players during real gated
  AI Aim movement.
- Preserve the full-field 1.0× pass so zoom cannot prevent acquisition or
  remove the same-frame fallback.
- Select 1.0×, 1.5×, or 2.0× automatically from the selected detection size.
- Keep all Makcu movement, STOP, disconnect, and generation guarantees.
- Publish Overlay boxes in the original centered 320-by-320 coordinate space.
- Keep unrelated base detections visible while replacing only the selected box
  with an associated refined box.
- Add no model, profile, training path, or runtime dependency.

## Non-goals

- Zooming the Overlay window or magnifying rendered pixels.
- Changing the physical centered 320-by-320 DXCam capture region.
- Running adaptive zoom for Overlay-only viewing, idle AI inference, or Test
  3s.
- Adding a user-controlled zoom slider, toggle, profile, or persisted zoom
  state.
- Predictive tracking, target lead, weapon-specific tuning, or model changes.
- Running two inference workers or queueing frames for later refinement.

## Approved behavior

Adaptive zoom is eligible only when all of these conditions are true:

- Makcu is connected.
- Master is armed.
- AI Aim is one of the selected movement sources.
- The normal Trigger and optional Modifier gate is active.
- No Test 3s mode is active.
- The current AI service generation is live.

Overlay visibility alone never enables the second pass. Releasing Trigger or
Modifier, disabling Master or AI Aim, STOP, disconnect, Test 3s, AI restart,
and shutdown all publish a false zoom-gate snapshot immediately. The next
worker observation uses one pass, and any refined result already in flight is
discarded after the non-cancelable inference call returns.

## Architecture

### Pure zoom module

Add `ai_zoom.py` as a hardware-free and Tk-free module. It owns the following
pure behavior:

- immutable `ZoomTransform` data containing the source crop's left, top, size,
  and zoom factor;
- class-specific zoom-factor selection;
- square crop placement and boundary clamping inside a 320-by-320 source;
- NumPy bilinear resizing back to RGB uint8 320-by-320;
- mapping refined detections and target points back into source coordinates;
- association of a mapped refined target with its base-pass seed;
- replacement of only the selected detection in the base detection frame.

`ai_zoom.py` may depend on NumPy and the immutable records in
`ai_targeting.py`. It does not import Tkinter, DXCam, ONNX Runtime, Makcu, or UI
state.

### Zoom-factor selection

The selected accepted detection from the base analysis determines the factor.
Its aim point must be no farther than 96 pixels from `(160, 160)`.

| Selected class | Detection height | Factor |
| --- | ---: | ---: |
| Head | 18 px or less | 2.0× |
| Head | 19-32 px | 1.5× |
| Player | 64 px or less | 2.0× |
| Player | 65-112 px | 1.5× |
| Either | Above its class threshold | 1.0× |

Boundary comparisons use the floating-point detection height without early
rounding. Unsupported classes are ineligible. A missing base target or
selected index always produces 1.0×.

### Crop and mapping

A 1.5× pass uses a 213-pixel square source crop; a 2.0× pass uses a 160-pixel
square. The crop is centered on the base target aim point and shifted as
needed, without changing its size, so every edge stays within `[0, 320]`.
The crop is bilinearly resized with NumPy to an owned contiguous RGB uint8
320-by-320 frame.

For a crop with source size `S` and origin `(L, T)`, model coordinates map back
as:

```text
source_x = L + model_x * S / 320
source_y = T + model_y * S / 320
```

All four detection edges and the selected aim point use the same transform.
Mapped boxes are clamped to `[0, 320]`; invalid or empty boxes are discarded.

### Association and refinement

The base selected detection is the seed. Each axis of its box is expanded on
both sides by the greater of 12 pixels or 20 percent of that axis's original
size. A mapped refined candidate is associated only when its aim point lies
inside this expanded box.

- A Player seed may refine to an associated Player or Head.
- A Head seed may refine only to an associated Head.
- Normal confidence filtering and head-first target selection remain in force.
- If multiple compatible candidates remain, the existing previous-target and
  nearest-target rules select among them.

On success, the mapped refined target becomes the movement target. The base
detection tuple remains intact except that its selected element is replaced by
the mapped refined selected detection. The selected index remains unchanged,
so the Overlay keeps all unrelated base boxes and emphasizes the refined box.
The published sequence and capture timestamp remain those of the base frame.

If no compatible refined target survives, the complete base analysis is
published without alteration.

### AI service integration

Extend `AiService.start` with a thread-safe boolean zoom-gate provider in
addition to the existing immutable `AimSettings` provider. The provider has a
safe false default for callers that do not request adaptive zoom.

For each latest captured frame, the generation-safe worker:

1. Runs the normal detector pass and base analysis.
2. Reads the zoom-gate provider.
3. Selects a factor from the base selected detection.
4. When the factor is greater than 1.0, rechecks generation, stop, and gate;
   builds the crop; and runs the same detector session a second time.
5. Rechecks generation, stop, and gate after the second inference.
6. Maps, associates, and composes the refined analysis, or uses the base
   analysis unchanged.
7. Atomically publishes one target and one detection-frame snapshot.

The capture loop always reads the newest DXCam frame. It never retains a zoom
request for a later frame. Published FPS continues to mean completed published
frames per second, not the number of individual model calls.

### UI gate snapshot

`JitterApp` owns one boolean adaptive-zoom gate under its existing short AI
snapshot lock. UI and queued service events update it on every relevant Master,
source, Trigger/Modifier, Test, connection, STOP, AI-error, and shutdown
transition. The value is true only for the approved normal gated state.

The AI worker receives a provider that reads only this boolean under the short
lock. It never reads Tk variables, `TriggerGate`, service objects, or widget
state from its worker thread.

### Runtime status

Add a `ZOOM` metric to the Motion page's `AI RUNTIME` card. It displays
`1.0×`, `1.5×`, or `2.0×`. `AiService` emits a generation-safe `zoom` event
only when the published factor changes. UI event epoch filtering prevents a
stale generation from changing the metric.

The display returns to `1.0×` on gate release, fallback, STOP, disconnect,
runtime stop, error, and shutdown. Zoom status, transforms, refined detections,
and gate state are runtime-only and are not written to `config.json`. Schema 5
does not change.

## Overlay behavior

The existing Overlay remains a centered, click-through, no-focus,
capture-excluded 320-by-320 window. It does not magnify its canvas.

- Base-pass boxes unrelated to the selected target remain visible.
- A successful refined detection replaces the selected base box at its mapped
  source coordinates and keeps the selected thick outline.
- A failed refinement leaves every base box unchanged.
- The configured color and `Head Boxes` filter apply after composition.
- With `Head Boxes OFF`, a refined Head remains valid for movement but its box
  is hidden, matching the existing display-only filter contract.

## Performance and cancellation

Ordinary frames use one inference. Eligible distant-target frames use two
sequential calls on the same detector session. Local probing measured roughly
10-14 ms per call on DirectML, so a refined frame is expected to take roughly
20-28 ms before preprocessing overhead.

There is no parallel detector session and no refinement queue. A DirectML call
already in progress cannot be preempted, but STOP and disconnect continue to
cancel Makcu movement through the existing independent movement barrier. A
result from an obsolete generation, stopped service, or released zoom gate is
discarded before publication.

## Error handling

- Base capture, model, contract, or inference errors retain the current
  generation-safe AI failure behavior.
- The first crop, resize, second-inference, mapping, or composition exception
  is logged with diagnostics. Adaptive refinement is then disabled for that AI
  generation, the UI returns to `1.0×`, and subsequent frames continue through
  the base 1.0× path.
- A refinement miss, low-confidence result, or association rejection is a
  normal fallback, not an error and not a reason to disable refinement for the
  generation.
- Overlay rendering errors retain the existing fail-closed Overlay behavior.
- No exception path may keep a stale zoom factor visible or publish a partial
  target/frame pair.

## Testing strategy

### Pure tests

- Every class-specific factor threshold and exact boundary.
- Center-distance eligibility at and beyond 96 pixels.
- 1.5× and 2.0× crop sizes, centering, and all four edge clamps.
- Bilinear resize output shape, dtype, ownership, and deterministic pixel
  values on a hand-derived fixture.
- Coordinate and box mapping, clamping, and invalid-box removal.
- Expanded association margin boundaries.
- Player-to-Head refinement, Head-to-Player rejection, and fallback.
- Selected-index preservation while unrelated base boxes remain unchanged.

### AI service tests

- Gate false and ineligible targets perform exactly one inference.
- Gate true with eligible 1.5× and 2.0× targets performs exactly two ordered
  inferences on the same captured frame.
- Successful refinement atomically publishes mapped target and composed frame.
- Miss, low confidence, and association rejection publish the base analysis.
- Gate release, STOP, restart, and stale generation after the second call drop
  the refined result.
- First refinement exception disables refinement once, logs once, emits 1.0×,
  and preserves later base inference.
- FPS counts published frames rather than model calls.
- Zoom events emit only on factor transitions and obey generation ownership.

### UI tests

- The zoom provider is true only for connected, Master-armed, AI-selected,
  normal Trigger/Modifier-active movement.
- Jitter-only, Overlay-only, Test 3s, release, STOP, disconnect, hotkey disable,
  AI error, and shutdown expose false immediately.
- The `ZOOM` metric follows current generation events and resets to `1.0×` on
  every terminal transition.
- No worker accesses Tk or Tk variables.
- Existing Overlay color, Head Boxes, source matrix, fallback, and fail-closed
  tests remain green.

### Verification and live acceptance

Run the repository-required compile, full unit suite, dependency imports,
DirectML self-check, and distribution review. Do not run Nuitka without a
separate explicit packaging request.

With the connected Makcu device, verify:

- near targets stay at 1.0×;
- medium distant targets transition to 1.5×;
- very small targets transition to 2.0×;
- mapped refined Overlay boxes remain aligned with the real target;
- unrelated base boxes remain visible;
- refine miss falls back to 1.0× without stopping movement;
- release, STOP, hotkey disable, disconnect, and shutdown stop movement
  immediately and reset Zoom;
- Overlay-only viewing and Test 3s never perform adaptive refinement;
- combined Jitter plus AI movement retains the existing one-report behavior.

## Success criteria

- Adaptive refinement runs only in the approved gated state.
- Every published refined coordinate is in the original 320-by-320 space and
  associated with the selected base target.
- Same-frame fallback always exists and never queues excess movement or stale
  inference.
- Zoom state is generation-safe, runtime-only, and accurately represented in
  the UI.
- All automated verification passes and live Makcu/Overlay acceptance confirms
  alignment and immediate cancellation.
