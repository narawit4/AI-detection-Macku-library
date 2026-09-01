# Realtime Detection Pipeline Design

## Goal

Reduce avoidable capture-to-overlay and capture-to-motion latency without
pretending that ONNX inference itself can run at 1,000 FPS. The existing
display-derived capture cadence remains capped at 240 FPS, while the existing
movement servo continues to consume the latest eligible target at 1,000 Hz.

## Approved runtime flow

1. DXCam remains the only capture runtime and continues to return an owned RGB
   `uint8` frame through `get_latest_frame(copy=True)`. Request DXCam's capture
   timestamp in the same call and do not make a second full-frame copy when the
   returned array is already owned and C-contiguous. A defensive copy remains
   mandatory for an injected/nonconforming array that is borrowed or
   non-contiguous.
2. `CapturedFrame` carries the optional source capture timestamp as the final
   field so existing positional test and integration constructors remain
   compatible. Production DXCam frames always carry a finite, non-negative
   timestamp. Injected legacy frames with no timestamp fall back to the
   service's observation time.
3. `AiService` publishes the source capture timestamp, not the later
   post-read/inference time. A malformed or future timestamp fails closed so a
   target cannot appear fresh indefinitely. Capture, detector, Adaptive Zoom,
   and publication continue to belong to exactly one generation worker.
4. The overlay checks the latest publication at the display-derived capture
   cadence (capped at 240 checks per second). Tk's integer delay uses
   `max(1, int(1000 / capture_fps))`, with 120 FPS as the invalid-value
   fallback. The overlay redraws only when the detection publication,
   fresh/stale state, runtime HUD tuple, or immutable style changes.
5. A frame crossing the existing 150 ms freshness boundary triggers exactly
   one clearing redraw. Continued stale polls do not redraw duplicates.
6. The existing 1,000 Hz absolute-deadline movement servo remains unchanged.
   It consumes the newest published target through existing time-based
   microsteps, discards it after 150 ms, sends no zero deltas, and never queues
   catch-up movement.

## Preserved behavior

- `Center 320` remains the startup default and `Full Display` remains the only
  alternative runtime capture mode.
- Base inference, same-frame Adaptive Zoom refinement, nearest current-frame
  target selection, generation barriers, STOP/disconnect behavior, and model
  switching contracts do not change.
- Overlay projection, box/HUD customization, click-through and capture
  exclusion behavior do not change.
- No prediction, tracking, interpolation of detections, additional inference
  workers, frame queues, model changes, dependencies, persisted runtime state,
  external-model copying, downloads, or packaging changes are introduced.

## Acceptance

- An owned contiguous frame returned by DXCam is reused as the
  `CapturedFrame.pixels` object; a borrowed/non-contiguous frame is copied once
  into owned C-contiguous storage.
- Both target and detection snapshots preserve the source capture timestamp.
- A source timestamp older than 150 ms is stale even if inference just
  completed.
- A 240 FPS runtime cadence schedules overlay checks every 4 ms; 120 FPS uses
  8 ms; invalid cadence falls back to 8 ms.
- Repeated polls of the same fresh publication draw once, a new sequence draws
  once, and crossing into stale state draws once to clear it.
- All hardware-free tests and the repository verification commands pass.
