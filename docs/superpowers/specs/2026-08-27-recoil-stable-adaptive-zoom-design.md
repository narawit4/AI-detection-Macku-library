# Recoil-Stable Adaptive Zoom Design

Date: 2026-08-27

## Context

Adaptive AI Zoom improves distant-target localization with a full-field base
pass followed by an optional target-centered 1.5x or 2.0x pass. Live Makcu
testing showed a remaining failure mode while firing: weapon recoil and the
optional Jitter source move the scene between captured frames. The Overlay can
then flicker, disappear, or jump to a different box. A 2.0x crop is especially
sensitive because its 160-pixel field of view is narrower than the 213-pixel
1.5x crop.

Adding 3.0x would make this failure mode worse and add inference cost. The
approved solution is temporal stability and hysteresis around the existing
1.0x, 1.5x, and 2.0x factors. The fixed model, centered 320-by-320 capture,
DirectML provider, optimized NumPy resize, base-pass acquisition, and mapped
Overlay coordinates remain unchanged.

## Goals

- Prevent 2.0x refinement from repeatedly entering while recoil or Jitter is
  moving the selected base target.
- Pause only the AI movement component when a newly acquired target or a large
  target jump has not yet been confirmed.
- Keep Jitter movement active while AI confirmation is pending.
- Preserve the latest honest Overlay frame without holding a stale AI movement
  target.
- Re-enter 2.0x automatically after the target is stable and the recoil
  cooldown has expired.
- Preserve STOP, Trigger/Modifier release, disconnect, shutdown, and generation
  cancellation guarantees.
- Add no model, dependency, profile, training path, or persisted setting.

## Non-goals

- Adding 3.0x, multi-tile search, a third inference call, or another detector
  session.
- Predicting recoil, compensating weapon-specific patterns, or leading moving
  targets.
- Holding or extrapolating a missing target for movement.
- Changing confidence thresholds, class priority, association radius, crop
  geometry, or mapped detection composition.
- Adding a UI control or configuration field for stability constants.
- Running stability-gated zoom for Overlay-only viewing or Test 3s.

## Fixed constants

The first implementation uses fixed internal constants:

- Stable target displacement: at most `18.0` pixels between consecutive base
  target aim points, measured by Euclidean distance.
- Stable confirmation: `2` consecutive observations of the same target class
  within the displacement boundary.
- Recoil cooldown: `0.100` seconds.

The exact `18.0` boundary is stable; `18.01` is unstable. The cooldown is
expired when the worker clock is greater than or equal to `cooldown_until`.
These values are runtime policy, not schema 5 configuration.

## Architecture

### Pure stability state

`ai_zoom.py` owns an immutable, hardware-free, and Tk-free
`ZoomStabilityState`. It records:

- the previous base-pass `TargetSnapshot`, or `None`;
- the number of consecutive stable base observations;
- the monotonic `cooldown_until` timestamp.

Pure functions consume the current state, current base target, and injected
monotonic time, then return a new state. The state is local to one AI service
generation and is never persisted or shared with the UI.

The state machine observes only base-pass targets. Refined coordinates do not
feed the stability calculation, preventing refinement precision or a failed
second pass from being misclassified as scene movement.

### Observation rules

For each current base target:

1. A missing target clears the previous target, sets the stable count to zero,
   and extends cooldown to at least `now + 0.100` seconds.
2. A first target after no target starts at stable count one and extends
   cooldown to at least `now + 0.100` seconds.
3. A target with a different class, or with displacement greater than `18.0`
   pixels from the previous base target, is an unstable acquisition. It becomes
   the new previous target, starts at stable count one, and extends cooldown to
   at least `now + 0.100` seconds.
4. A same-class target within `18.0` pixels increments the stable count, capped
   at two.
5. A refinement miss or association rejection extends the zoom cooldown to at
   least `now + 0.100` seconds and resets the stable count to zero while keeping
   the latest base target available for the next comparison.

Extending cooldown uses `max(existing_cooldown_until, now + 0.100)` so a later
unstable observation cannot shorten an existing cooldown.

### AI movement confirmation

AI movement is confirmed when the current base target has stable count two.
Movement confirmation is independent of the 100 ms zoom cooldown:

- On first acquisition, class change, or a displacement above 18 pixels, the
  current detection frame remains publishable to the Overlay, but the
  published movement target is `None` for that frame.
- On the next same-class observation within 18 pixels, AI movement resumes.
- No previous target position is held, extrapolated, or sent to Makcu while
  confirmation is pending.
- In combined Jitter+AI mode, the AI component contributes `(0, 0)` while
  unconfirmed; the Jitter component continues through the existing combined
  motion engine.
- A missing base target continues to publish no movement target, matching the
  existing contract.

At normal capture and inference cadence, confirmation adds approximately one
frame, typically 20-35 ms. It never waits for the 100 ms zoom cooldown before
resuming ordinary base or 1.5x AI movement.

### Stability-limited zoom factor

The existing class/height/center rules still compute a requested factor. The
stability layer produces the factor used for the optional second pass:

| Requested factor | Stability state | Applied factor |
| ---: | --- | ---: |
| 1.0x | Any | 1.0x |
| 1.5x | Any base target | 1.5x |
| 2.0x | Stable count 2 and cooldown expired | 2.0x |
| 2.0x | Otherwise | 1.5x |

The first small-target acquisition therefore uses the wider 1.5x crop. A
stable target can use 2.0x after cooldown. A recoil jump immediately caps the
next refinement at 1.5x. The `ZOOM` metric displays the factor actually
published, not the original requested factor.

No stability state creates a second pass when the existing zoom gate is false.
There remain at most two detector calls for one captured frame.

### AI service integration

`AiService` creates a fresh stability state inside each worker generation. For
each captured frame, the worker:

1. Runs base inference and base analysis using a separate per-generation
   `selection_previous` target as the existing 48-pixel
   selection-association origin, then updates that target from the current base
   analysis regardless of whether AI movement is confirmed.
2. Reads the existing generation-safe zoom gate.
3. If the gate is false, resets stability state and follows the one-pass base
   path.
4. If the gate is true, observes the base target and derives movement
   confirmation.
5. Computes the requested factor, limits it through the stability state, and
   runs at most one refinement pass using the applied factor.
6. On refinement success, composes the mapped result as today.
7. On a normal refinement miss, publishes the same-frame base detection frame,
   extends cooldown, resets confirmation, and publishes no movement target for
   that frame; it never revives a previous target.
8. If the current acquisition is unconfirmed, publishes the current detection
   frame with a `None` movement target after composition or fallback.
9. Atomically publishes one movement target and one detection-frame snapshot.

The stability state's previous base target is used only for displacement and
class confirmation. The separate `selection_previous` target may guide only
the next frame's existing target association. Neither is published for
movement while confirmation is pending. This keeps Overlay-only selection
continuity unchanged when the zoom gate is false and avoids losing a candidate
merely because one confirmation frame intentionally publishes `None`.

## Runtime and generation safety

- Trigger/Modifier release, Mouse5 disable, STOP, disconnect, AI stop, error,
  restart, and shutdown immediately expose a false zoom gate and reset the UI
  metric to `1.0x` through the existing path.
- A worker observation of a false gate replaces its local stability state with
  the initial state before another gated acquisition can occur.
- State belongs to the worker stack. Obsolete generations cannot mutate or
  publish state into a newer worker.
- All existing generation and stop checks before and after the non-cancelable
  DirectML call remain in place.
- No stability decision delays the independent Makcu cancellation barrier.
- Overlay-only and every Test 3s source matrix remain outside the zoom gate and
  never activate stability refinement.

## Overlay behavior

The Overlay continues to display only the latest published detection frame in
original 320-by-320 source coordinates:

- An unconfirmed target does not hide current boxes; it only removes the AI
  movement target.
- A successful refined box replaces only its associated selected base box.
- A refinement miss leaves the full same-frame base detection tuple visible.
- No old box or target is held across a missing frame.
- Color, line width, selected emphasis, capture exclusion, and `Head Boxes`
  filtering remain unchanged.

## Error handling

- Base capture, model, contract, or inference errors keep the existing
  generation-safe failure behavior.
- A crop, resize, second-inference, mapping, or composition exception still
  disables refinement for that generation, logs diagnostics, and continues the
  base path.
- Disabling refinement resets stability state and publishes `1.0x`.
- A normal miss or association rejection is not logged as an exception. It
  extends cooldown and uses honest same-frame base detection-frame fallback
  with no movement target for that frame.
- State-transition helpers reject no runtime inputs; they normalize only
  through the explicit missing, class-change, distance, and clock rules above.

## Performance

- Ordinary frames still use one inference.
- Eligible frames still use at most two sequential calls on the same detector
  session.
- Unstable 2.0x requests use the existing optimized 213-to-320 resize and 1.5x
  inference; they do not add another fallback call in the same frame.
- Stability operations are constant-time scalar comparisons and immutable
  record replacement. They do not allocate image-sized arrays or add a queue.
- Published FPS continues to count published frames, not detector calls.

## Files and responsibilities

- `ai_zoom.py`: immutable stability state, pure observation, cooldown, factor,
  and confirmation decisions.
- `ai_service.py`: per-generation state ownership, ordered application around
  the existing base/refinement flow, and movement-target suppression.
- `tests/test_ai_zoom.py`: exact pure state boundaries and transitions.
- `tests/test_ai_service.py`: inference-count, publication, fallback,
  cancellation, and generation integration tests.
- `README.md` and `AGENTS.md`: concise recoil-stability runtime behavior and
  test inventory only if implementation changes make existing text incomplete.

No UI, settings schema, model, capture, detector, Makcu, Overlay, or packaging
module requires a new public control or dependency.

## Testing strategy

### Pure stability tests

- First acquisition begins at stable count one, starts cooldown, and withholds
  movement.
- Same-class displacement exactly 18 pixels confirms; 18.01 pixels restarts
  confirmation and cooldown.
- A class change is unstable even inside the distance boundary.
- A missing target clears movement confirmation and previous target.
- Requested 1.5x applies immediately.
- Requested 2.0x is capped to 1.5x before confirmation and during cooldown.
- Requested 2.0x applies at the exact 100 ms cooldown boundary when confirmed.
- A refinement miss resets stability count, extends but never shortens
  cooldown, and retains only the latest base target for comparison.
- Fresh initial state contains no held target, cooldown, or confirmation.

### AI service tests

- The first eligible small-target frame runs a 1.5x second pass, publishes its
  frame, and publishes no AI movement target.
- The second stable frame resumes AI movement; 2.0x remains capped until the
  cooldown boundary.
- After cooldown, a stable small target runs exactly one 2.0x second pass.
- A recoil jump above 18 pixels immediately downgrades 2.0x to 1.5x and
  suppresses AI movement for the new acquisition frame.
- Combined-mode target suppression yields no AI component while leaving the
  Jitter component unaffected through existing combined-motion tests.
- Refinement miss publishes current base boxes, never an old movement target,
  and causes the next 2.0x request to remain capped.
- Gate release, STOP, restart, and stale generations cannot publish old state,
  factor, target, or frame.
- No case performs more than two detector calls for one captured frame.
- Existing refinement exception and FPS accounting behavior remains intact.

### Complete verification

Run the repository's canonical compile, full unit suite, runtime imports,
DirectML model self-check, and distribution review. Do not run Nuitka.

Hardware acceptance uses the connected Makcu in both AI-only and Jitter+AI
modes:

- While firing, a target jump pauses only AI movement for one confirmation
  frame and does not snap to a stale position.
- Jitter continues during AI confirmation.
- The Overlay may follow the actual shaken frame but must not hold an old box.
- Unstable zoom falls to 1.5x or 1.0x, then returns to 2.0x after stability and
  cooldown.
- Trigger release, Mouse5, STOP, disconnect, reconnect, and shutdown retain
  immediate cancellation and reset behavior.
- Post-interaction logs contain no stability, zoom, generation, Overlay, or
  Makcu errors.

## Acceptance criteria

- Recoil and Jitter do not cause an immediate AI movement snap to an
  unconfirmed target more than 18 pixels from the previous base target.
- AI movement resumes after two stable same-class base observations without
  waiting for the 100 ms zoom cooldown.
- 2.0x does not resume until both confirmation and cooldown requirements are
  satisfied.
- Current-frame Overlay fallback remains visible while stale movement is never
  published.
- Maximum detector calls remain two per captured frame.
- All automated and connected-hardware checks pass without new dependencies,
  configuration, or packaging changes.
