# Combined AI Aim, Jitter, and Detection Overlay Design

Date: 2026-08-27
Status: Approved in chat

## Context

Jitter currently exposes one mutually exclusive `Jitter / AI Aim` mode and one
enabled state. The selected production motion engine runs through one
generation-safe Makcu worker while the configured Trigger and optional Modifier
are held. `AiService` owns the fixed centered 320-by-320 DXCam capture and the
bundled ONNX detector, but it publishes only the selected target needed for AI
movement.

This design replaces exclusive modes with independently selectable Jitter and
AI sources, combines both outputs safely in one Makcu motion worker, and adds an
optional red detection-box overlay. This is an explicit new design decision
that supersedes the earlier AI Aim non-goals for combined movement and overlays.
The fixed model, capture size, target-selection policy, dependency restrictions,
and clean-room boundary remain unchanged.

## Goals

- Add independent `Jitter` and `AI Aim` source-selection buttons.
- Allow Jitter alone, AI Aim alone, or both sources together.
- When both sources are selected, add their current two-dimensional deltas and
  send at most one final Makcu report per composite tick.
- Continue Jitter normally when AI has no valid target.
- Keep one master armed state controlled by the existing UI action and global
  hotkey.
- Add an independent overlay toggle that can run detection without arming or
  moving the Makcu device.
- Draw red rectangles around every accepted player and head detection in the
  centered 320-by-320 capture area, with the selected target emphasized.
- Preserve immediate STOP, disable, disconnect, source-change, Test Run, and
  shutdown cancellation guarantees.
- Keep all Tk work on the main thread and all capture, inference, and blocking
  Makcu work off the Tk event loop.
- Add no runtime dependency or alternate model.

## Non-goals

This feature does not add labels, confidence text, configurable overlay colors,
full-screen or multi-monitor capture, overlay resizing, target history, trails,
auto-click, triggerbot, silent aim, button masking, profiles, tray behavior,
training, alternate models, or alternate inference runtimes. It does not add
Pillow, Pystray, Torch, Ultralytics, OpenCV, or another GUI toolkit.

## Runtime state model

The UI owns four independent, non-persisted Boolean states:

- `jitter_selected`: Jitter contributes to production and Test Run movement.
- `ai_selected`: AI Aim contributes to production and Test Run movement.
- `master_armed`: selected production sources may move while the Trigger and
  optional Modifier gate is active.
- `overlay_visible`: detection boxes are displayed; this never grants movement
  authority.

All four states start `False` on every application launch regardless of the
saved configuration or a legacy saved mode. Held buttons, Moving state, test
state, AI targets, detections, snapshots, FPS, provider, and runtime status also
remain non-persisted.

The existing global hotkey and the primary master button call the same master
toggle operation. Arming is rejected with an actionable footer message when no
source is selected or Makcu is disconnected. Disarming immediately cancels the
active motion generation but does not change source selections or an
independently visible overlay.

STOP is stronger than ordinary master disable. It immediately cancels movement
and Test Run, sets `master_armed` and `overlay_visible` to `False`, clears held
button state, hides and clears the overlay, and releases AI runtime demand that
is no longer needed. STOP preserves `jitter_selected` and `ai_selected` as inert
source choices so the user can deliberately re-arm them later.

Closing the window cancels every runtime, destroys the overlay, closes services,
and exits the process. It does not preserve any of the four runtime states.

## Source selection while armed

Source buttons may be changed while Master is armed. A source change never
modifies an already running motion worker in place:

1. Cancel the current motion generation through the existing final stop
   barrier.
2. Update the immutable source-selection snapshot.
3. Start or stop AI runtime according to the new demand.
4. If Master remains armed, Makcu is connected, the Trigger/Modifier gate is
   still active, and at least one source remains selected, reserve a new motion
   generation using the new source set.

The generation handoff uses the existing deferred-action and exact retiring
source checks so a stale terminal event cannot start or stop a newer worker.
Removing the last selected source automatically disarms Master after cancelling
movement.

## Composite movement

### Pure composite engine

A new `combined_motion.py` module owns pure orchestration of the existing
`MotionEngine` and `AimMovementEngine`. It has no Tkinter or Makcu dependency.
It receives an immutable source selection, current motion settings, the latest
AI target snapshot, current AI settings, and a monotonic timestamp.

Each enabled component keeps its existing semantics:

- Jitter retains its pulse timing, paired-direction behavior, acceleration
  limits, per-axis fractional accumulation, and discard-not-queue rule.
- AI retains one consumption per target sequence, stale-target rejection,
  dead zone, Aim Strength, smoothing, acceleration limit, per-axis fractional
  accumulation, AI Max Step, and discard-not-queue rule.

For each composite tick, disabled or currently idle components contribute
`(0, 0)`. When AI has no fresh valid target, AI contributes `(0, 0)` and does
not affect Jitter timing. The composite engine adds the two integer component
deltas axis by axis, clamps the final X and Y independently to the Makcu report
range `[-127, 127]`, and discards any excess. Final-clamp excess is never stored
or added to a later tick. A zero final pair produces no controller call.

The scheduler wakes for the earliest next due component without busy-spinning.
AI may react to a newly published inference snapshot before the next Jitter
pulse; Jitter pulses remain scheduled by their own configured rate. Multiple
component outputs that are due in the same worker iteration are combined into
one report.

### Makcu ownership

`MakcuService` remains the sole owner and caller of the Makcu controller. It
adds one composite start entry point accepting source, motion-settings,
target-snapshot, and AI-settings providers. Jitter-only and AI-only operation
use the same composite path with one component disabled, so all three source
combinations share identical generation, cancellation, event, disconnect, and
cleanup behavior.

Only one production or Test Run motion worker may be active. The service never
runs independent Jitter and AI controller workers. STOP returning continues to
guarantee that the stopped generation cannot begin another Makcu movement
report.

## Detection snapshots

`ai_targeting.py` adds an immutable detection-frame snapshot containing:

- the monotonically increasing inference sequence;
- the monotonic capture timestamp;
- a tuple of accepted immutable `Detection` values; and
- the index of the selected detection, or `None` when no target is selected.

Accepted detections are only class `player` (`0`) and `head` (`7`) values that
pass the current configured Confidence threshold and existing finite-coordinate,
box-geometry, and capture-bound validation. The selection index refers to that
filtered tuple. Target preference and association remain unchanged: heads are
preferred, players are fallback targets, and the existing selected aim point is
used for movement.

`AiService` atomically publishes both the latest `TargetSnapshot` and latest
detection-frame snapshot under its short lock after each current-generation
inference. It exposes separate snapshot-provider methods for movement and
overlay consumers. Snapshots are replaced, never queued. Stop, failure, close,
or generation invalidation clears both values. A result from an obsolete
generation can update neither value nor the UI.

The existing 150-millisecond target freshness limit also applies to overlay
frames. Consumers clear rather than reuse an expired snapshot.

## AI runtime demand

Capture and inference run whenever at least one current demand exists:

- `overlay_visible`;
- `master_armed and ai_selected`; or
- an AI or combined `Test 3s` is loading or active.

Demand reconciliation is centralized in the UI lifecycle controller. Adding the
first demand starts `AiService`; removing the final demand stops it and clears
snapshots. Removing AI movement demand while Overlay remains visible leaves the
same AI generation running. Hiding Overlay while armed AI remains selected also
leaves the same generation running. This prevents avoidable model reloads and
preserves generation safety.

Overlay-only operation does not require Makcu connection and cannot call a
Makcu movement API. Selecting AI while Master is off also does not by itself
start inference; Overlay may still create independent demand.

## Overlay component

A new `overlay.py` module defines a detection overlay with a small interface for
show, render, clear, hide, and close. It owns a borderless Tk `Toplevel` and
canvas, and every one of its methods is called on the Tk main thread.

The window is exactly 320 by 320 pixels and is positioned at the center of the
Windows primary display, matching DXCam output index `0` and the fixed capture
region. It is always on top, excluded from task switching, does not activate or
take keyboard focus, and is click-through. It uses the current theme-independent
red overlay color. Accepted detections use a thin outline; the selected
detection uses the same red with a thicker outline. It draws no text, fill,
crosshair, capture-region border, or stale box.

The UI polls the latest detection-frame snapshot with `after(...)`. Rendering
replaces all prior canvas rectangles in one main-thread update. No worker
accesses Tk, and no overlay event is queued per detection. Empty, absent, or
older-than-150-millisecond snapshots clear the canvas immediately while leaving
the requested overlay window visible.

After the Tk window handle exists, `overlay.py` uses Windows extended styles
equivalent to layered, transparent-to-input, tool-window, and no-activate
behavior. It then requires `SetWindowDisplayAffinity` with
`WDA_EXCLUDEFROMCAPTURE` so DXCam cannot capture the red rectangles and feed
them back to the detector. If any required click-through or capture-exclusion
setup fails, the overlay window is destroyed, `overlay_visible` becomes
`False`, the footer shows a concise actionable error, and detailed Windows
diagnostics go to `app.log`. AI movement may continue; if Overlay was the only
AI demand, `AiService` is stopped.

Hiding, STOP, and shutdown clear the canvas before withdrawing or destroying
the window. Cleanup failures are logged and never block STOP or process exit.

## UI design

The fixed-size English Focused Dashboard remains one page. Advanced content
continues to scroll inside the existing outer dimensions, and the red STOP
button remains visible when Advanced Settings is expanded.

The existing exclusive Mode combobox is removed. Its control area gains two
independent source-selection buttons whose visible states are `Jitter OFF/ON`
and `AI Aim OFF/ON`. The existing primary enable action becomes
`Enable Selected` while disarmed and `Disable Selected` while armed. It remains
the visible master action controlled by the global hotkey. Primary cyan actions
keep dark readable text; secondary controls keep light readable text.

The AI settings section gains `Overlay OFF/ON` and retains AI runtime status,
provider, FPS, Confidence, Aim Strength, Smoothing, and Max Step. Jitter and AI
settings are both available in the Advanced scroll area regardless of source
selection so settings can be prepared before arming. Numeric controls retain
slider plus exact-value inputs.

The runtime status distinguishes Disabled, Armed, Moving, and Testing as it does
today. Source button states and AI status make the active combination explicit;
the footer reports concise source-change, load, overlay, and error messages.

During `Test 3s`, source-selection buttons and the Test button are disabled.
STOP remains enabled. The Overlay toggle remains independent, but STOP hides it.

## Master hotkey and sounds

The configured global hotkey remains edge-triggered once per press and invokes
the same master toggle as the UI. It never changes source selections or Overlay.
When no source is selected, the hotkey leaves Master off and shows the same
actionable message as the UI master action.

Existing enable and disable sounds follow successful Master state changes. A
rejected arm, source-selection change, Overlay toggle, or direct UI source
button does not play the master hotkey cue. Stale queued hotkey events remain
discarded through the existing epoch mechanism after STOP, disconnect,
reconnect, or shutdown.

## Test Run behavior

`Test 3s` uses the currently selected production sources:

- Jitter selected alone runs Jitter.
- AI selected alone runs AI Aim.
- Both selected run the composite engine.
- Neither selected rejects the test with an actionable footer message.

Test Run requires Makcu, bypasses Trigger and Modifier temporarily, and remains
interruptible by STOP, disconnect, shutdown, or a service error. It preserves
the pre-test Master and source-selection states. If AI is required, the
three-second interval begins only after `AiService` is Ready; load or capture
failure aborts the test. A successful duration completion restores the prior
Master state. STOP or disconnect cancels without re-arming. Overlay demand is
independent and is not created automatically by Test Run.

## Disconnect, failure, and recovery

Makcu disconnect immediately cancels motion and Test Run, disarms Master,
clears held buttons, and advances the motion and hotkey epochs. It preserves
inert source selections. Overlay may remain visible and continue inference
because it is independent of Makcu. If Overlay is off, AI runtime loses its
movement demand and stops.

An AI runtime error clears AI and detection snapshots, turns `ai_selected` and
`overlay_visible` off, hides the overlay, and cancels the current composite
motion generation. If Jitter remains selected and Master was armed, the UI
starts a fresh Jitter-only generation when the Trigger/Modifier gate is active;
otherwise Master is disarmed. The detailed exception is logged while the footer
states that AI stopped and Jitter remains available.

An overlay setup or rendering error turns off and destroys only Overlay. It
does not deselect AI or stop AI movement unless Overlay was the final AI runtime
demand. A Makcu movement error retains the existing centralized emergency-stop
behavior. Cleanup errors are logged without delaying cancellation.

The previously observed physical cable auto-reconnect defect is outside this
feature's implementation scope; the new state transitions must not conceal or
worsen it.

## Configuration migration

The settings schema increments from version 3 to version 4 because exclusive
mode is removed from the current document shape. Schema 4 stores motion, AI,
bindings, hotkey, preset, theme, and sound settings but does not store `mode`,
source selections, Master, or Overlay.

Loading schema 3 validates and preserves all supported settings but ignores its
legacy `mode` for runtime selection; all runtime states still start off.
Schemas 1 and 2 continue through their existing safe migrations before applying
the same runtime-off rule. A newer unsupported schema remains untouched, runs
with safe in-memory defaults, and prohibits saving according to the existing
contract. Atomic temporary-file, flush, `fsync`, backup, and replace behavior is
unchanged.

## Threading and cancellation invariants

- Tk widgets, Tk variables, `Toplevel`, and canvas access stay on the main
  thread.
- AI capture and inference stay on one daemon worker with stop events and
  generation identity.
- Makcu movement stays on one daemon worker with one controller owner.
- UI event sinks continue to marshal worker events through the bounded queue.
- Source and settings providers return immutable snapshots under short locks.
- STOP, master disable, disconnect, source change, and shutdown signal
  cancellation immediately rather than waiting for a normal movement interval.
- Stale AI, service, motion, and hotkey events cannot mutate a newer lifecycle.
- No excess composite movement is queued for a later report.

## Testing strategy

Development follows test-driven development. Every behavior change receives a
failing test that is observed before the minimal implementation is added.

Pure and service tests cover:

- Jitter-only, AI-only, and combined composite steps;
- AI `(0, 0)` behavior without a valid target while Jitter continues;
- per-component timing and state plus final axis sum and `[-127, 127]` clamp;
- final excess discard, fractional behavior, acceleration limits, and no zero
  controller calls;
- one Makcu worker, exact generation terminals, STOP barrier, disconnect,
  source replacement, and cleanup;
- immutable accepted-detection snapshots, Confidence filtering, selected index,
  stale clearing, stop, error, and obsolete-generation rejection; and
- AI runtime demand reconciliation without unnecessary restart.

Overlay tests use injected Tk/window and Win32 adapters where platform calls
would otherwise prevent deterministic assertions. They cover centered geometry,
red rectangle coordinates, selected thickness, full-frame replacement, empty
and stale clearing, always-on-top/click-through/no-activate styles,
`WDA_EXCLUDEFROMCAPTURE` failure, hide, STOP, and close. Tests assert that
worker threads never call Tk-facing overlay methods.

UI tests cover every source combination, launch-off defaults, rejected empty
selection, Master button and hotkey parity, source changes while armed, removal
of the last source, independent Overlay demand, Test Run combinations, AI
failure fallback to Jitter, disconnect with Overlay remaining visible, STOP,
shutdown, sound cues, and non-persistence of runtime state. Configuration tests
cover schema 3-to-4 migration, schema 4 serialization, malformed values, atomic
writes, and unsupported future-schema protection.

After implementation, run the repository verification commands with
`combined_motion.py` and `overlay.py` added to compilation. Nuitka is not run.
Hardware verification with the connected Makcu covers Jitter-only, AI-only,
combined movement, no-target Jitter continuation, Trigger/Modifier, Master
hotkey, STOP, disconnect, Test Run, and shutdown. On-screen verification covers
box alignment, selected thickness, click-through behavior, stale clearing, and
confirmation that the overlay is absent from DXCam capture.

## Files and documentation

Expected implementation changes include:

- create `combined_motion.py` for pure source composition;
- create `overlay.py` for the main-thread Windows/Tk overlay;
- modify `ai_targeting.py`, `ai_service.py`, `makcu_service.py`, `ui.py`,
  `settings.py`, and their focused tests;
- update compile and distribution metadata only where new imported modules must
  be named explicitly;
- update `README.md` for independent sources, Master, Test Run, and Overlay;
  and
- update `AGENTS.md` only where exclusive-mode and no-overlay guidance conflicts
  with this approved design decision, plus the planned module and verification
  lists.

The fixed model, license materials, pinned runtime dependencies, capture size,
and packaging confirmation policy remain unchanged. Ordinary feature work does
not run Nuitka or modify generated output directories.
