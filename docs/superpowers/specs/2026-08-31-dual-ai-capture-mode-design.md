# Dual AI Capture Mode Design

Date: 2026-08-31

## Context

The full-primary-display change intentionally replaced Jitter's original
centered 320-by-320 DXCam capture. That replacement matched the earlier written
scope, but it did not match the clarified product intent: the original fast
center capture must remain available, and full-display detection must be an
additional selectable mode.

This design supersedes only the sole-capture-mode decision in
`2026-08-31-full-primary-display-detection-design.md`. Its native-frame
letterboxing, source-coordinate targeting, Adaptive Zoom, overlay projection,
generation safety, and fixed 1,000 Hz motion design remain authoritative.

## Goals

- Restore the original physical centered 320-by-320 capture as the startup
  default on every launch.
- Add a runtime `Full Display` capture choice without running a second AI
  worker or detector.
- Switch capture mode while normal AI Aim or Overlay inference is active by
  restarting only the AI generation.
- On a successful live switch, preserve source selections, Master state,
  Trigger/Modifier state, Jitter motion, and Overlay visibility.
- Immediately invalidate the previous mode's target, detections, zoom state,
  and queued AI events so coordinates from two capture geometries cannot mix.
- Keep the full-screen click-through Overlay window in both modes. In center
  mode, project boxes only into the physical centered 320-by-320 screen region;
  keep the HUD at its configured screen corner.
- Preserve model contracts, target selection, response curve, smoothing,
  clamping, 1,000 Hz servo behavior, model switching, and cancellation rules.

## Non-Goals

- No simultaneous center and full-display inference.
- No automatic mode selection, region tracking, movable crop, custom crop
  dimensions, multi-monitor capture, or second detector session.
- No capture-mode persistence, configuration-schema change, profile, download,
  model copy, or additional bundled model.
- No new HUD metric or main-dashboard AI runtime readout.
- No attempt to make the bundled 320 model retain full-display detail that is
  absent after letterboxing; external compatible 640 models remain a
  runtime-only user choice.

## User-Facing Modes

The AI Aim Settings section gains one compact, read-only `Capture Mode`
combobox beside `Target Area`:

- `Center 320` maps to the internal value `center_320`.
- `Full Display` maps to the internal value `full_display`.

`Center 320` is initialized explicitly and independently of persisted
configuration before the controls and runtime services are created. Capture
mode is runtime-only and is never read from or written to `config.json`.

The combobox is available while idle and during established normal AI or
Overlay inference. It is disabled during `Test 3s`, model validation/model
rollback, capture-mode restart, shutdown, and any other transition in which a
second lifecycle change would be ambiguous. The handler also checks these
guards so a programmatic event cannot bypass the disabled UI state.

## Capture Boundary

`jitter_app/ai/capture.py` owns the two physical DXCam regions and validates the
mode. Pure helpers remain independently testable:

```text
center_320  -> centered_region(output_width, output_height, 320)
full_display -> full_output_region(output_width, output_height)
```

The centered-region calculation restores the exact prior behavior: floor the
left and top offsets so a 320-by-320 square is centered on the primary output,
and reject an output smaller than 320 in either dimension. Full display remains
`(0, 0, output_width, output_height)`.

Each successful read returns one immutable `CapturedFrame` value containing an
owned, contiguous RGB `uint8` image together with capture geometry from the
same camera generation:

- primary-output width and height;
- capture-region left and top;
- capture-region width and height;
- selected capture mode.

The returned image dimensions must equal the capture-region dimensions. A
missing frame remains `None`; malformed pixels or inconsistent geometry fail
the current AI generation through the existing runtime error path. Geometry is
paired with the image at the capture boundary so neither the service nor the
Overlay infers screen placement from image dimensions.

DXCam still uses output index 0, RGB, NumPy processing, a two-frame maximum
buffer, and the runtime display-derived capture cadence capped at 240 FPS.

## AI Service and Generation Lifecycle

`AiService.start()` accepts a validated capture-mode keyword. The mode is
copied into the worker arguments for that generation; a later UI edit cannot
mutate an already running worker. The default capture factory creates exactly
one `DxcamCapture` for that mode. Hardware-free injected factories retain a
narrow equivalent seam and provide captured images with explicit geometry.

Restarting a generation recreates its detector and capture resource under the
existing ownership model, but does not restart the Tk application, Makcu
service, hotkey watcher, motion worker, or Overlay window.

For each frame, the service sends only its RGB image to the detector and uses
the captured image width and height for targeting and Adaptive Zoom. It adds
the primary-output dimensions and capture origin to the published detection
snapshot. The target snapshot needs no screen origin because movement remains
relative to the center of the captured frame.

Changing capture mode follows this sequence on the Tk thread:

1. Reject the event if a protected lifecycle transition is active.
2. Record the new runtime-only mode and disable the combobox for the restart.
3. Invalidate the AI event epoch and signal the current AI generation to stop.
4. Clear target/detection publication, ready/provider/FPS state, and zoom state.
5. If AI demand still exists, start one new generation using the same committed
   model and the newly selected capture mode.
6. Re-enable the control after the exact current-generation ready or terminal
   event, and leave it enabled immediately when there is no AI demand.

The motion worker is not restarted. During the short AI restart it observes no
AI target; selected Jitter therefore continues through the unchanged combined
motion path, while AI-only movement emits nothing. No target or delta from the
old capture geometry is queued or reused.

Successful restart preserves Master, source selections, held-button state, and
Overlay visibility. A synchronous or worker startup failure uses the existing
AI failure policy: hide Overlay, deselect AI Aim, retain eligible Jitter, and
disarm Master only when no selected working source remains. There is no capture
mode rollback or repeated retry.

STOP, disconnect, source removal, hotkey disable, model change, and shutdown
retain their existing higher-priority cancellation behavior.

`Test 3s` uses the capture mode selected when the test starts and holds that
mode fixed until the test ends. Model candidate and rollback generations use
the currently selected capture mode. STOP and AI runtime failures do not reset
the user's runtime selection; only a new process launch restores `Center 320`.

## Targeting and Adaptive Zoom

Both modes use the same current-frame, nearest-target algorithm:

```text
capture center = (frame_width / 2, frame_height / 2)
```

For `Center 320`, this is exactly `(160, 160)`, preserving the original target
choice, distance normalization, response curve, and movement behavior. For
`Full Display`, the already implemented native-frame center and normalization
remain unchanged.

Adaptive Zoom always crops within the current captured image. Its base pass and
optional same-frame refinement therefore remain 320-source-space operations in
center mode and native-display-source-space operations in full mode. A capture
mode restart resets zoom confirmation and cooldown by creating a new AI
generation. Refinement eligibility and the one-additional-inference bound do
not change.

## Overlay Projection

The Overlay remains one primary-display-sized, click-through,
capture-excluded window in both modes. Detection snapshots carry enough
geometry to map captured-frame coordinates to the Overlay canvas:

```text
output_x = capture_left + source_x
output_y = capture_top  + source_y
canvas_x = output_x * canvas_width  / primary_output_width
canvas_y = output_y * canvas_height / primary_output_height
```

DXCam capture pixels and region pixels are one-to-one, so no additional crop
scale is introduced. Full display uses a zero origin and fills the canvas.
Center mode uses the centered region origin, so detection boxes occupy only the
same physical 320-by-320 area that DXCam captured. Clipping is performed first
in captured-frame coordinates and then to the Overlay canvas.

HUD placement, offsets, color, font, metric filters, freshness, selected lock,
box labels, player/head visibility, and runtime-only customization remain
unchanged. The HUD stays at the configured screen corner rather than moving
into the center capture region.

## UI State and Presentation

The new field shares one compact two-column row with `Target Area`; it does not
add a new dashboard section or restore AI runtime metrics to the main UI. Its
popup uses the shared themed combobox palette. The collapsed AI section summary
includes the selected capture label along with its existing model, target, and
strength information, within the existing truncation limit. No duplicate
status panel is added.

When the user selects a different mode, the footer reports a concise transition
such as `Switching AI capture to Full Display...`, followed by the existing
ready or actionable error state. Selecting the already active value is a no-op.

## Configuration and Packaging

There is no schema change. `AppConfig`, the atomic store, backup behavior, and
schema 1-6 compatibility remain untouched. Tests must prove that capture mode
is absent from serialized configuration and returns to `Center 320` in a new
app instance.

The dependency set, bundled model, model validation contract, Nuitka data
options, licenses, and release materials do not change. Ordinary implementation
verification does not run Nuitka.

## Error Handling and Threading

- Tk variables and widgets remain main-thread only.
- Capture mode is copied into immutable worker startup arguments.
- Stale ready, FPS, zoom, error, target, and detection results are rejected by
  the existing UI epoch and service generation checks.
- Capture startup/read/geometry failures close the camera exactly once and
  enter the current AI error policy.
- A disabled combobox is not the safety boundary; handler guards prevent mode
  changes during Test, model transitions, shutdown, or an existing mode switch.
- STOP and shutdown signal cancellation immediately and never wait for capture
  or inference cadence.

## Test Strategy

Development follows RED-GREEN-refactor in these slices:

1. Capture helpers and DXCam boundary: exact centered/full regions, default
   center mode, geometry/image agreement, malformed frames, cleanup, and
   display-too-small behavior.
2. Service: mode copied per generation, default factory construction, geometry
   publication, stale-result rejection, restart clearing, and unchanged
   detector/zoom call bounds.
3. Overlay: centered 320 placement on landscape and portrait outputs, full
   display projection, canvas scaling, clipping, stale frames, and HUD staying
   at the configured screen corner.
4. UI: compact two-column field, startup default, runtime-only behavior, popup
   palette, live switch preserving Master/source/Overlay/Jitter, AI-only quiet
   interval, protected-transition guards, failure policy, STOP, and shutdown.
5. Regression: centered mode reproduces prior 320 targeting and Adaptive Zoom;
   full display retains native-frame behavior; model selection and 1,000 Hz
   motion tests remain unchanged.

After focused tests, run all canonical AGENTS verification commands. Hardware
acceptance additionally checks both modes on a connected Makcu device, live
switching while Trigger is held, Overlay alignment, click-through and capture
exclusion, combined Jitter+AI behavior, STOP, reconnect, and shutdown.

## Documentation Changes

Implementation updates `AGENTS.md` and `README.md` to describe two selectable
capture modes, the Center 320 startup default, runtime-only selection, live
generation restart, and mode-specific Overlay mapping. The earlier full-display
spec remains historical and is explicitly superseded only where it declares
full display to be the sole mode.
