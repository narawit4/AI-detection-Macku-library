# AI Aim Mode Design

Date: 2026-08-26
Status: Approved in chat

## Context

Jitter currently owns a Makcu controller and produces an interruptible paired-pulse
motion while the application is enabled and the configured Trigger and optional
Modifier are held. This design adds an exclusive AI Aim mode that captures a
small region at the center of the primary display, detects targets with the
approved `all_games_320.onnx` model, and sends closed-loop relative movement
through the existing Makcu lifecycle.

The Eventuri project at
`C:\Users\User\Downloads\Eventuri-AI-MAKCU-0.1.0` was inspected as a reference.
Its source is not copied. Jitter will use only the approved model binary and a
clean-room implementation of the required capture, inference, targeting, and
movement boundaries.

## Goals

- Add mutually exclusive `Jitter` and `AI Aim` operating modes.
- Run `all_games_320.onnx` through ONNX Runtime DirectML without Torch,
  Ultralytics, or OpenCV runtime dependencies.
- Prefer the nearest valid `head` detection and fall back to the upper portion
  of the nearest valid `player` detection.
- Move only while Jitter is enabled and its Trigger plus optional Modifier gate
  is active.
- Preserve immediate STOP, disable, disconnect, mode-change, Test Run, and
  shutdown cancellation guarantees.
- Keep all Tk access on the main thread and all blocking capture, inference, and
  Makcu calls off the Tk event loop.
- Package the model with a distributable executable while publishing the full
  corresponding source under AGPL-3.0.

## Non-goals

This feature does not add auto-click, triggerbot, silent aim, button masking,
NDI input, overlays, debug windows, profiles, raw serial access, WindMouse,
Bezier movement, AI training, or support for arbitrary model formats. It does
not import Eventuri configuration, GUI code, device code, or user data.

## Licensing and distribution

The ONNX metadata identifies the model as an Ultralytics YOLOv10 model licensed
under AGPL-3.0. The user has approved distributing Jitter and its complete
corresponding source under AGPL-3.0.

Implementation will:

- add the repository-level `LICENSE` containing AGPL-3.0;
- add `THIRD_PARTY_NOTICES.md` with model provenance, SHA-256, and relevant
  dependency notices;
- copy the approved model to `models/all_games_320.onnx` without modifying it;
- record the source model SHA-256 as
  `6B9157D6419F9DBC40D2DCECCC33A3387078C86F1C5872EDA544B174FF48499C`;
- document that each distributed executable must be accompanied by access to
  the exact corresponding Jitter source; and
- update, but not run, `gen.bat` so an explicitly requested build includes the
  model and required DirectML runtime files.

## Dependencies

The runtime dependency set adds these packages to the existing pinned
requirements:

- `onnxruntime-directml==1.24.4`
- `dxcam==0.3.0`
- `numpy==2.5.2`

The implementation must not add Torch, Ultralytics, OpenCV, MSS, CustomTkinter,
or Eventuri dependencies. DirectML is the preferred execution provider;
`CPUExecutionProvider` is the explicit fallback when DirectML session creation
fails.

## Model contract

The approved model has this fixed contract:

- input name `images`;
- input tensor `float32[1,3,320,320]`;
- RGB channel order, CHW layout, and values normalized to `[0,1]`;
- output name `output0`;
- output tensor shape `[1,300,6]`;
- each valid output row represents `x1, y1, x2, y2, confidence, class_id` in
  capture-region coordinates; and
- relevant classes are `player` with id `0` and `head` with id `7`.

The detector validates names, shapes, finite coordinates, confidence, class id,
and non-empty box geometry. A model-contract mismatch is a fatal AI runtime
error, never a movement command.

## Components

### `ai_capture.py`

`DxcamCapture` owns one DXCam camera. It discovers the primary output size,
computes a centered 320-by-320 region, requests RGB frames, and returns an owned
NumPy array. Capture resources are created and closed on the AI worker, not the
Tk thread. A missing or malformed frame is skipped without reusing an old image.

### `ai_detection.py`

`OnnxDetector` creates an ONNX Runtime session, preferring
`DmlExecutionProvider` and falling back to `CPUExecutionProvider`. It validates
the model contract, converts one RGB frame to a normalized contiguous NCHW
tensor, runs inference, and converts valid output rows to immutable `Detection`
values. It has no Tkinter or Makcu dependency.

### `ai_targeting.py`

This pure module defines immutable `Detection`, `TargetSnapshot`, and
`AimSettings` values plus target-selection and movement logic.

Target selection first filters by configured confidence. If one or more heads
remain, it selects a head; only when no head remains may it select a player. A
head aims at the box center. A player aims at 20 percent of box height below
the top edge. Initial selection chooses the candidate whose aim point is nearest
the 160-by-160 capture center.

For stability, a same-class candidate whose aim point is within 48 pixels of the
previous aim point remains selected. A newly detected head replaces a retained
player immediately. If no associated candidate remains, selection returns to
the nearest candidate. A snapshot older than 150 milliseconds is invalid.

`AimMovementEngine` consumes each snapshot sequence at most once. It computes
the error from capture center, applies a fixed 1.5-pixel radial dead zone,
multiplies by Aim Strength, and calculates each smoothed axis as
`previous + (desired - previous) * (1 - Smoothing)`. It then limits change to
6 Makcu counts per axis per consumed snapshot, clamps each axis to Max Step and
the Makcu report range, and preserves fractional remainders. Values beyond the
current clamp are discarded rather than queued. When several same-class
candidates are within the 48-pixel association radius, the one nearest the
previous aim point is retained.

### `ai_service.py`

`AiService` owns model loading, capture, inference, target selection, FPS
measurement, and the latest immutable target snapshot. It exposes start, stop,
close, status, and snapshot-provider operations but never accesses Tk or a Makcu
controller.

Starting creates a new generation and one daemon worker. The worker initializes
the model and capture resources, emits `loading`, then emits `ready` with the
active provider. It continuously replaces the latest snapshot; detections are
never queued. Stop sets the current event immediately and invalidates the
generation. Results produced by an obsolete generation are ignored. Worker
events are marshalled to the existing UI queue.

### `makcu_service.py`

`MakcuService` remains the only owner and caller of the Makcu controller. It adds
an AI-target motion entry point that uses the existing motion generation,
connection generation, cancellation event, final stop barrier, and event
semantics. The AI mover reads the newest `TargetSnapshot` and immutable
`AimSettings`, passes them through `AimMovementEngine`, and calls
`controller.move(x, y)` only after the same final generation and stop checks as
normal Jitter motion.

Normal Jitter and AI target motion can never be active simultaneously. STOP
returning guarantees that the stopped generation cannot begin another Makcu
movement report.

### `settings.py`

The configuration schema is incremented. `AppConfig` adds the selected mode and
an immutable AI settings record. Older supported schemas migrate to `Jitter`
mode with safe AI defaults. Future schemas remain untouched according to the
existing save prohibition.

AI settings and validation ranges are:

- Confidence: default `0.35`, range `0.05` through `0.95`;
- Aim Strength: default `0.35`, range `0.05` through `2.00`;
- Smoothing: default `0.65`, range `0.00` through `0.95`; and
- Max Step: default `20`, range `1` through `127`.

Held buttons, selected targets, target snapshots, FPS, runtime provider, AI
status, and Moving state are never persisted.

### `ui.py`

The existing fixed-size interface adds a `Jitter / AI Aim` mode selector and a
compact AI section containing status, FPS, Confidence, Aim Strength, Smoothing,
and Max Step. Numeric AI settings use the existing slider plus exact-value input
pattern. Advanced content continues to scroll within the existing window, and
the red STOP button remains visible.

AI runtime state is shown as `Stopped`, `Loading`, `Ready (DirectML)`,
`Ready (CPU)`, or `Error`. Detailed exception text goes only to `app.log`; the
footer shows a short actionable message.

## Lifecycle and data flow

1. The app always launches Disabled.
2. Selecting AI Aim does not move the device.
3. Enabling AI Aim while Makcu is connected starts `AiService`; the model loads
   and centered capture/inference begins so a fresh target is ready without
   trigger-press startup latency.
4. Trigger plus optional Modifier activation starts the Makcu AI motion worker.
5. Each new detection replaces the latest snapshot. The mover consumes a fresh
   sequence once, transforms its error through the pure movement engine, and
   sends at most one clamped two-dimensional report for that snapshot.
6. Trigger or Modifier release stops only Makcu movement; capture and inference
   remain armed while AI Aim stays enabled.
7. STOP, disable, disconnect, mode change, or shutdown immediately signal both
   movement and AI runtime cancellation and invalidate their generations.
8. Disconnect uses the existing emergency-stop path and does not automatically
   re-arm AI after reconnect.

`Test 3s` uses the currently selected production engine. In AI Aim mode it
requires a connected Makcu, starts AI runtime loading asynchronously when it is
not already Ready, and begins the three-second interval only after Ready. A load
or capture error aborts the test. It temporarily bypasses Trigger and Modifier
and remains interruptible by STOP, disable, disconnect, mode change, or
shutdown. If AI runtime was not armed before the test, the test stops it while
restoring the prior application state.

## Error handling

- Missing, unreadable, corrupt, or contract-incompatible model: enter AI Error,
  do not start AI movement, leave Jitter mode available.
- DirectML initialization failure: log diagnostics and retry session creation
  with CPU only; expose the fallback in status.
- DXCam initialization or capture failure: enter AI Error, clear the latest
  snapshot, and send no movement.
- Individual malformed detections or empty frames: skip them and continue; do
  not reuse an expired target.
- Inference failure after Ready: clear the target, enter AI Error, and stop the
  active AI movement generation.
- Makcu movement failure or disconnect: use the existing centralized service
  event and emergency-stop behavior.
- Cleanup failures: log them without blocking shutdown.

## Testing

All automated tests remain hardware-free through injected capture, session,
clock, engine, and controller fakes.

Unit tests cover:

- centered capture-region calculation and frame ownership;
- RGB-to-normalized-NCHW preprocessing;
- model contract validation and `[1,300,6]` parsing;
- invalid coordinates, confidence, class ids, and boxes;
- head priority, player fallback aim point, nearest selection, association,
  immediate player-to-head switch, and stale snapshot rejection;
- dead zone, Aim Strength, smoothing, acceleration limiting, clamping,
  fractional accumulation, discard-not-queue behavior, and one consumption per
  snapshot sequence;
- DirectML provider selection and CPU fallback;
- AI service status, FPS, generation invalidation, stop, and cleanup;
- mutual exclusion of Jitter and AI motion;
- Trigger and Modifier gating, Test 3s bypass, STOP, disconnect, hotkey disable,
  mode change, and shutdown;
- configuration defaults, migration, validation, atomic save, and unsupported
  future-schema protection; and
- UI queue marshalling, displayed state, mode-dependent controls, and concise
  errors.

Implementation verification extends the repository commands to compile the new
modules, runs the complete unittest suite, imports `makcu`,
`onnxruntime`, `dxcam`, and `numpy`, and verifies the copied model hash. Hardware
verification requires a connected Makcu and an approved on-screen test target.
Nuitka packaging is not run unless explicitly requested.

## Documentation changes

README will explain AI mode prerequisites, the fixed model, DirectML/CPU status,
controls, safety stops, AGPL source availability, and source/build commands.
AGENTS.md will be updated only where its previous no-AI scope statement conflicts
with this explicit approved design decision; unrelated repository guidance stays
unchanged.
