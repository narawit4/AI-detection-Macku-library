# Conservative AI Tracker, Response Curve, and Adaptive Cadence Design

Date: 2026-08-27

## Context

Live Makcu testing exposed two related AI Aim problems. First, two same-class
detections that pass within the existing 48-pixel association radius can trade
identity immediately. The three-frame replacement guard never runs because
every nearby candidate is treated as the confirmed target. A deterministic
crossing simulation reproduces the switch: after target A and target B cross,
nearest-point association follows B even though the lock began on A.

Second, the Makcu motion worker polls at 240 Hz, but `AimMovementEngine` emits
movement only once for each new inference sequence. Every poll that sees the
same snapshot returns `(0, 0)`. The resulting large reports separated by empty
polls feel rigid even when Smoothing is high. DXCam capture is independently
fixed at 120 FPS and does not follow the primary display refresh rate.

The approved design replaces point-only association with a conservative
temporal box tracker, turns AI movement into a time-based servo, adds a
five-point distance-to-speed response curve, and derives capture/servo cadence
from the primary display. When identity is uncertain, AI movement pauses
instead of guessing. Jitter, Overlay visibility, the fixed model, DirectML,
and every existing cancellation boundary remain independent.

## Goals

- Keep the confirmed person through close passes and crossings when geometry
  and recent motion identify that person clearly.
- Publish no AI movement while two plausible detections are too ambiguous to
  distinguish.
- Require stable observations before switching to a genuinely different
  target, while allowing Jitter to continue.
- Produce small AI reports on every eligible servo tick instead of one burst
  per inference result.
- Let users shape near-to-far response with a safe five-point graph while
  retaining Confidence, Aim Strength, Smoothing, and Max Step.
- Use the primary display refresh rate for adaptive capture and motion cadence
  without making total movement speed depend on refresh rate.
- Preserve stale-frame rejection, fractional accumulation, clamping, excess
  discard, immediate STOP/disconnect/gate cancellation, and generation safety.

## Non-goals

- Persistent person identities across AI service generations, trigger
  sessions, application restarts, or long detector outages.
- Face recognition, appearance embeddings, optical flow, model changes, or a
  general-purpose multi-object tracking dependency.
- Continuing to move through an ambiguous or missing target.
- User-facing tracker thresholds, per-game profiles, training, prediction
  lead, recoil compensation, or alternate capture/model runtimes.
- Variable-refresh synchronization, present-event hooks, or changing the
  display mode.
- Packaging as part of ordinary implementation verification.

## Runtime data flow

```text
primary display Hz -> RuntimeCadence -> DXCam target FPS
                                  \-> Makcu AI servo interval

base detections -> conservative tracker -> confirmed target -> adaptive zoom
                         |                       |
                         | ambiguous             v
                         +-> all Overlay boxes   240-style time servo
                                                   |
response curve + Aim settings ---------------------+-> Makcu delta
```

The tracker sees only full-field base detections. Refined detections never
change identity state. The Overlay always receives current accepted boxes;
only a currently confirmed, unambiguous base box receives selected emphasis
and can seed refinement or movement.

## Conservative temporal tracker

### Ownership and state

A new `ai_tracking.py` module owns immutable, pure tracking state and geometry.
It imports the existing immutable detection/snapshot records from
`ai_targeting.py` and has no Tk, DXCam, ONNX Runtime, or Makcu dependency.

One `TrackerState` belongs to one `AiService` worker generation and records:

- the last confirmed base detection and target snapshot;
- the preceding confirmed aim point and timestamps used for velocity;
- ambiguity/recovery state;
- a pending replacement candidate and consecutive count; and
- the last time at which the confirmed track was unambiguous.

No tracker state is persisted. A new AI generation always starts empty, and an
obsolete worker cannot publish its state into a current generation.

### Candidate construction and class priority

Each observation first applies the existing Confidence threshold and accepts
only class 7 heads and class 0 players. Initial acquisition retains the current
head-first rule: use heads when any accepted head exists, otherwise use
players. Within that class, distance from the 320-by-320 frame center remains
the initial preference.

Once a track exists, same-class plausible matches are considered before any
replacement class. A head does not silently become an unrelated player, and a
different head cannot bypass replacement confirmation merely because it is
close. If no plausible same-class match exists, ordinary head-first selection
chooses only a pending replacement.

### Prediction and plausibility

The tracker predicts the next aim point from the two latest confirmed points.
Velocity is bounded to 800 capture pixels per second, and extrapolation is
capped at 100 ms. Ambiguous frames never update the confirmed box or velocity.

A same-class candidate is plausible only when:

- its aim point is within `max(48 px, 1.5 * confirmed box diagonal)` of the
  predicted point; and
- its area is between 0.4 and 2.5 times the confirmed box area.

These gates scale close-target motion with box size while retaining the old
48-pixel minimum for small boxes.

Each plausible candidate receives a lower-is-better score:

- 60% predicted-point distance normalized by the plausibility radius;
- 25% `1 - IoU` against the confirmed box; and
- 15% bounded absolute log-area change.

Geometry is the only identity evidence available from the approved detector.
The score therefore does not claim certainty when two boxes explain the old
track equally well.

### Ambiguity and confirmation

If the two best plausible candidates differ in score by at most 0.15, the
frame is ambiguous. The exact 0.15 boundary is ambiguous. During ambiguity:

- the published movement target is `None`;
- `selected_index` is `None`, so no current box is shown as selected;
- adaptive refinement does not run;
- all current accepted base boxes remain available to the Overlay;
- confirmed geometry and velocity are not updated; and
- Jitter continues through normal combined-motion composition.

After ambiguity, the same predicted track must be the clear best match for two
consecutive observations before AI movement resumes. A changed recovery
candidate resets the recovery count. This prevents a one-frame split from
deciding identity at the crossing boundary.

If the confirmed track has no clear plausible match for 150 ms, its predictive
hold expires. A replacement must then remain the same class, within 18 pixels
of its preceding pending aim point, and clear of ambiguity for three
consecutive observations. The third observation becomes confirmed. The
original track returning before expiry cancels replacement. Missing detections
never publish a held coordinate.

The 150 ms hold is measured from the last clear confirmed observation and is
expired when elapsed time is greater than or equal to 0.150 seconds. The
18-pixel pending boundary is inclusive.

Initial acquisition is immediate only when one candidate is clearly preferred
and no competing candidate lies within the ambiguity margin. An ambiguous
initial choice uses the same three-observation confirmation rule. Initial
candidates are scored by center distance divided by the frame's
center-to-corner distance; a best-to-runner-up gap of 0.15 or less is
ambiguous, matching the tracked-candidate margin.

### AI service integration

`AiService` replaces `TargetLockState` with one `TrackerState` per worker. For
each captured frame it:

1. runs full-field base inference;
2. passes base detections and the current settings to the tracker;
3. publishes the tracker's current accepted detection tuple;
4. runs adaptive zoom only when the tracker returns a confirmed target and
   selected base index and the existing zoom gate allows refinement;
5. composes a successful refinement exactly as today; and
6. atomically publishes the final same-frame detection snapshot and either the
   confirmed target or `None`.

Refinement failure keeps the existing same-frame base fallback. A base frame
that is ambiguous cannot be made unambiguous by the zoom pass, and zoom output
never feeds the next track observation.

## Adaptive display cadence

A new `display_timing.py` module isolates the Windows refresh query and pure
cadence validation. `query_primary_display_hz()` uses the current primary
display mode because DXCam is fixed to output index 0. The Win32 adapter is
injectable for hardware-free tests and catches unsupported/malformed results.

`RuntimeCadence` contains runtime-only values:

- `display_hz`: the validated detected rate, or `None` on fallback;
- `capture_fps`: `min(display_hz, 240)` for a valid rate;
- `servo_hz`: `clamp(2 * display_hz, 120, 480)` for a valid rate.

A display rate is valid only when finite and between 24 and 500 Hz inclusive.
Invalid or unavailable values produce `display_hz=None`, `capture_fps=120`,
and `servo_hz=240`. A valid reported value is rounded to its nearest integer
before the capture/servo policies are applied.

`JitterApp` obtains one immutable cadence at startup, with injection available
to tests. Its default factories pass `capture_fps` to `DxcamCapture` and
`servo_hz` to `MakcuService`/`CombinedMotionEngine`. Existing injected service
factories remain supported. Cadence is not recalculated during a run and is
never persisted.

Capture rate controls how often DXCam samples output 0. Detector throughput
may be lower; `get_latest_frame` continues to discard obsolete buffered frames
rather than building latency. Servo rate controls only the number of motion
integration opportunities. All speed and smoothing equations use measured
`dt`, so a higher refresh rate creates smaller, more frequent reports without
increasing intended displacement per second.

## Five-point response curve

### Settings contract

`AimSettings` gains an immutable `response_curve` tuple containing Y values at
fixed normalized distances `(0.0, 0.25, 0.50, 0.75, 1.0)`. The approved
default is:

```text
(0.00, 0.12, 0.35, 0.68, 1.00)
```

The first value is always exactly zero. All values are finite, lie in 0..1,
and are monotonically non-decreasing. Mapping input must contain exactly five
numeric values satisfying the complete contract; otherwise the entire curve
uses the default. Runtime UI edits enforce the same constraints before
replacing the immutable snapshot.

A pure monotone cubic Hermite interpolator evaluates the curve. It passes
through all five points, never overshoots the adjacent Y values, and handles
flat segments without division by zero. Inputs are clamped to 0..1.

### Time-based servo

`AimMovementEngine.step` continues to receive the latest snapshot, settings,
and monotonic time, but repeated snapshot sequences no longer return zero.
The engine records an estimated remaining target error, filtered velocity,
fractional reports, and the previous tick time.

On a fresh sequence, the observed `aim - center` replaces the estimated error.
On each later eligible servo tick, the engine:

1. derives normalized radial distance using the 320-by-320 center-to-corner
   distance;
2. evaluates the response curve;
3. computes `curve_distance = curve(normalized_distance) * hypot(160, 160)`,
   then `reference_step = min(Max Step, curve_distance * Aim Strength)` and
   desired speed `reference_step * 60` capture pixels per second along the
   remaining-error direction;
4. applies Smoothing as a `dt`-based time constant and applies a `dt`-scaled
   acceleration limit equivalent to the current six-pixel-per-60-Hz-step
   limit;
5. integrates velocity over actual `dt`, combines fractional residue, and
   emits a bounded integer Makcu report; and
6. subtracts only the integer movement actually reported from the estimated
   error.

The vector direction always points toward the estimated target. A report is
clamped so neither axis crosses past zero remaining error. Max Step remains a
per-report safety boundary, while the reference-rate velocity cap preserves a
comparable overall limit across servo rates.

Smoothing 0 responds immediately. For a positive value, the exact time
constant is `0.200 * (smoothing / 0.95) ** 2` seconds. The velocity filter uses
`alpha = 1 - exp(-dt / time_constant)`. The acceleration limit is 21,600
capture pixels per second squared, the time-scaled equivalent of changing a
60 Hz reference step by six pixels. These mappings use elapsed time, not tick
count, so they behave consistently at 120 through 480 Hz.

If the snapshot is `None`, older than 150 ms, changes generation through the
existing worker lifecycle, or reaches the 1.5-pixel dead zone, the engine
discards velocity, estimated error, and fractional residue immediately. No
unfinished response is queued for a future target. STOP, disconnect, source
change, hotkey disable, Trigger/Modifier release, and shutdown retain their
existing outer cancellation barriers.

## UI behavior

The Motion page adds one full-width `AIM RESPONSE CURVE` card below the
existing AI settings/runtime row inside the current scrollable content. The
outer window size and always-visible STOP area do not change.

The graph is a themed Tk Canvas with five fixed X positions. The zero-distance
point is visible but fixed at zero. The 25%, 50%, 75%, and 100% points can move
vertically. Dragging clamps a point between its neighboring Y values, keeping
the curve monotonic. The graph redraws from the same pure interpolation used
by the engine.

Four exact percentage entries accompany the draggable points. Invalid text,
out-of-range values, or monotonicity violations use the existing invalid-entry
style and do not replace the live snapshot. `Reset Curve` restores the
approved default. Valid drag or entry changes update the immutable settings
snapshot immediately and use the existing debounced configuration save.

The AI Runtime card adds a concise cadence line. A detected 144 Hz display
shows `DISPLAY 144 HZ · SERVO 288 HZ`; fallback shows
`DISPLAY AUTO · SERVO 240 HZ`. This is runtime status only.

All Canvas and Tk-variable access remains on the main thread. Theme changes
redraw the graph with shared palette colors, keyboard focus remains visible,
and exact entries provide a non-pointer editing path.

## Configuration schema 6

Schema 6 persists `ai.response_curve` and no cadence/tracker runtime state.
The JSON value is an exact five-element array, for example
`["0", "0.12", "0.35", "0.68", "1"]`; ordinary scalar AI controls remain
strings as today. Curve Tk variables are separate from the existing scalar
`ai_vars` mapping.
Schemas 1 through 5 retain their existing migration behavior and receive the
default curve in memory. A normal later save writes schema 6 through the
existing temporary-file, flush, `fsync`, backup, and atomic-replace path.

Malformed response curves use the complete safe default rather than partially
repairing a shape. A schema identifier newer than 6 still disables saving and
leaves the source file untouched. Selected sources, Master, held inputs,
targets, boxes, tracker state, ambiguity, display Hz, capture FPS, servo Hz,
velocity, fractions, and curve-render state are never persisted.

## Error handling and safety

- Refresh-query failures log detailed diagnostics and use the fallback cadence;
  they do not prevent startup or create an AI runtime error.
- DXCam/model/capture/inference failures retain the existing AI failure path.
- Invalid curve data never reaches the motion worker.
- Ambiguity, a normal tracker miss, or replacement confirmation is normal
  state and is not logged as an exception.
- Overlay-only inference can maintain tracker state but cannot create Makcu
  movement without the existing Master and Trigger/Modifier gate.
- Test 3s uses the production curve and servo but retains its existing trigger
  bypass and adaptive-zoom exclusion.
- Combined motion still clamps the final Jitter-plus-AI report to Makcu's
  two-dimensional range; wheel data is never used as a movement axis.
- Existing stop events, generation checks, move barrier, and stale callback
  suppression remain authoritative over every adaptive interval.

## Files and responsibilities

- `display_timing.py`: injectable Win32 refresh query and pure cadence policy.
- `ai_tracking.py`: immutable tracker state, geometry, candidate scoring,
  ambiguity, recovery, and replacement confirmation.
- `ai_targeting.py`: response-curve settings/validation/interpolation and the
  time-based AI movement servo.
- `ai_capture.py`: accept adaptive `target_fps` while retaining centered owned
  RGB frames and output index 0.
- `ai_service.py`: own one tracker per generation and integrate confirmed base
  selection with existing zoom/refinement publication.
- `combined_motion.py`: expose the injected AI servo cadence.
- `makcu_service.py`: construct/poll the combined engine at the injected
  cadence without weakening cancellation.
- `ui.py`: obtain/inject cadence, render/edit the curve, show runtime cadence,
  and publish immutable settings.
- `settings.py`: schema 6 migration and atomic response-curve persistence.
- `main.py`: no eager UI/AI import changes; ordinary startup continues through
  `JitterApp` and self-check isolation remains intact.
- `README.md` and `AGENTS.md`: document tracker ambiguity, curve semantics,
  adaptive cadence, schema 6, new modules, and updated verification commands.
- `tests/`: hardware-free pure, service, movement, UI, schema, cadence,
  cancellation, and integration coverage.

No requirement, model, license, overlay-native behavior, or packaging command
changes.

## Testing strategy

### Tracker tests

- Two same-class boxes crossing paths retain A through velocity prediction and
  never confirm B merely because B becomes closer to A's previous point.
- Equal or near-equal crossing scores produce ambiguity, `target=None`, and
  `selected_index=None` while preserving every accepted Overlay box.
- Exact score-margin, plausibility-radius, area-ratio, velocity-cap, prediction-
  horizon, and 150 ms hold boundaries are deterministic.
- Two clear recovery frames resume the original track; a changed recovery
  candidate resets the count.
- A missing/implausible original cannot publish stale movement, and a
  replacement becomes confirmed only on its third stable observation.
- Head-first initial and replacement behavior remains intact.
- Tracker state is immutable and independent across AI generations.

### Curve and servo tests

- Default and custom curves pass through all fixed points, are monotonic, and
  never overshoot; malformed curves use the complete default.
- Repeated use of one fresh snapshot yields multiple bounded microsteps rather
  than one nonzero report followed only by zeros.
- Equal elapsed time at 120, 288, and 480 Hz produces equivalent intended
  displacement within integer/fractional tolerance.
- Strength, Smoothing, Max Step, dead zone, stale age, acceleration, fraction,
  no-axis-overshoot, and reset boundaries remain deterministic.
- A new snapshot corrects remaining error and discards obsolete excess rather
  than appending a plan.
- `None`, stale data, STOP, disconnect, and gate release cannot leak a late
  microstep.

### Cadence, service, schema, and UI tests

- Win32 refresh success, malformed values, exceptions, 24/500 Hz boundaries,
  capture cap, servo floor/cap, and fallback are tested with injected APIs.
- DXCam receives the derived integral target FPS while retaining its fixed
  region/output/color/buffer contract.
- AiService crossing/ambiguity/recovery tests verify current-frame Overlay,
  no refinement while ambiguous, no stale movement, and at most two detector
  calls per captured frame.
- Combined/Makcu workers use the injected interval and retain immediate
  cancellation and final report clamping.
- Schema 1-5 migration, schema 6 round trip, malformed curve fallback, atomic
  backup, and unsupported-future no-overwrite behavior are covered.
- Graph layout, theme redraw, pointer drag, keyboard/exact entry, validation,
  Reset Curve, live snapshot replacement, debounced save, fixed window, scroll,
  and always-visible STOP behavior are covered without real hardware.

### Complete verification

Run the repository's canonical compile command, full unit suite, pinned runtime
imports, DirectML self-check, and distribution review. Do not run Nuitka.

Connected Makcu acceptance covers AI-only and combined Jitter+AI at the actual
detected display cadence:

- fire while two targets approach, overlap, cross, and separate;
- verify no pull toward the competing target during ambiguity;
- verify AI pauses while Jitter continues, then resumes the original track;
- verify a genuinely lost target requires three stable replacement frames;
- compare near, middle, and far curve edits for continuous microsteps without
  bursts, oscillation, drift, or overshoot;
- verify the displayed cadence matches the primary Windows display policy;
- verify Trigger/Modifier release, hotkey disable, STOP, disconnect, reconnect,
  Test 3s, and shutdown remain immediate; and
- inspect `app.log` for tracker, cadence, motion, generation, Overlay, AI, or
  Makcu errors.

## Acceptance criteria

- A close or overlapping competitor cannot become the confirmed target through
  same-class proximity alone.
- Ambiguous frames publish no AI movement and no selected Overlay index while
  retaining current detection boxes.
- The original target resumes only after two clear recovery observations; a
  replacement requires three stable observations.
- One fresh target snapshot produces bounded movement across multiple adaptive
  servo ticks until corrected, stale, canceled, or centered.
- Total intended speed is time-based and materially invariant across supported
  display/servo rates.
- The five-point graph is live, monotonic, exactly editable, resettable, themed,
  and persisted only as validated schema 6 AI settings.
- Display/capture/servo cadence follows the approved caps and safe fallback and
  remains runtime-only.
- Jitter, adaptive zoom limits, Overlay capture exclusion, fixed model/runtime,
  generation isolation, and immediate cancellation guarantees remain intact.
- All automated verification and connected-hardware acceptance checks pass with
  no new dependencies or packaging run.
