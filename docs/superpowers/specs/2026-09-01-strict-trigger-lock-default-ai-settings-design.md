# Strict Trigger Lock and Recommended AI Defaults Design

Date: 2026-09-01

Status: Approved in chat on 2026-09-01.

> **Binding target-selection supersession:** During an eligible raw-Trigger
> press, this design replaces stateless current-frame movement selection with
> the strict fail-closed lock below. Outside such a press, current-frame
> nearest selection remains available for Overlay visualization and initial
> acquisition. This is the only supersession of the existing clean-room AI Aim
> selection contract; model contracts, capture modes, movement composition,
> and cancellation behavior remain unchanged.

## Context

The current production path analyzes every base inference independently and
selects whichever accepted aim point is nearest the source-frame center. That
is responsive, but a second player crossing nearer to the crosshair can replace
the selected player immediately. Low Strength can make general assistance feel
weak while a maximized near-distance curve still makes that replacement feel
abrupt.

The requested behavior is stricter: acquire at most one target for one press
of the configured Trigger, follow only an unambiguous continuation of that
target, and never select a replacement until the Trigger is released and
pressed again. If continuity becomes uncertain, AI movement must stop instead
of guessing. Jitter may continue through the existing independent composition
path.

The detector does not expose a persistent person identifier. A `Detection`
contains box geometry, confidence, and class only. Therefore this design does
not promise perfect knowledge of physical identity through indistinguishable
occlusion or merged detector output. It does provide the testable guarantee
that the application performs no second acquisition and no observable
replacement after the first ambiguous or missing continuation in a Trigger
epoch.

The user also approved a balanced set of AI settings as the application
defaults. These defaults increase useful pull while using a progressive curve,
moderate smoothing, and a lower step cap to avoid abrupt direction changes.

## Goals

- Allow at most one target acquisition during one configured raw-Trigger
  press.
- Continue publishing movement only while exactly one plausible base-box
  continuation exists.
- Latch irreversibly into a lost state on a missing, non-unique, invalid, or
  ambiguous continuation.
- Require Trigger-up followed by Trigger-down before any new acquisition.
- Prevent stale pre-Trigger, pre-reset, or obsolete-generation targets from
  reaching Makcu.
- Preserve all accepted Overlay boxes while clearly reporting no lock after
  loss.
- Keep Jitter independent when AI has no locked target.
- Preserve Full Display, Center 320, Adaptive Zoom, Test 3s, model switching,
  immediate cancellation, and generation safety.
- Make the approved settings the defaults for new, missing, or malformed
  configuration values without overwriting valid schema-5 user choices.

## Non-goals

- Face recognition, appearance embeddings, ReID, optical flow, or a new model
  contract.
- Claiming guaranteed physical identity when the detector observations are
  indistinguishable.
- Holding or predicting movement through a missing or ambiguous base target.
- Automatically reacquiring after loss while the same Trigger press remains
  active.
- Persisting Trigger epochs, target boxes, lock state, target history, capture
  mode, model choice, or other runtime state.
- Adding user-facing tracker thresholds, profiles, downloads, training, or new
  dependencies.
- Migrating valid schema-5 settings or introducing schema 6.
- Running Nuitka as part of ordinary implementation verification.

## Recommended default settings

`AimSettings` and `DEFAULT_RESPONSE_CURVE` will use:

| Setting | Default |
| --- | ---: |
| Confidence | `0.25` |
| Aim Strength | `0.35` |
| Smoothing | `0.58` |
| Max Step | `18` |
| Curve at 0% | `0.00` |
| Curve at 25% | `0.16` |
| Curve at 50% | `0.38` |
| Curve at 75% | `0.68` |
| Curve at 100% | `0.95` |

`Center 320` remains the startup Capture Mode. The first curve point remains
fixed at zero, all points remain ordered and bounded, and Reset Curve restores
this complete new tuple.

The configuration schema remains 5:

- a missing or malformed scalar uses its new field default;
- a missing or malformed schema-5 curve uses the complete new curve default;
- schemas 1 through 4 use the complete new curve default;
- every valid explicit schema-5 value remains authoritative, including values
  equal to the old defaults; and
- schema 6 remains unsupported, saving stays disabled, and its source file is
  left byte-for-byte unchanged.

There is no automatic migration and no direct edit of local `config.json`.
This prevents the application from treating an intentional old value as an
untouched default.

## Terminology and guarantee

**Raw Trigger** is the configured Trigger button without applying the optional
Modifier. `TriggerGate.trigger_held` is the source of this state.

**Eligible press** is a Raw Trigger down edge that occurs while the normal AI
source is eligible: the device is connected, Master is armed, AI Aim is
selected, the app is not closing, and no Test 3s run owns movement. A press
that begins while ineligible cannot become eligible merely because Master or
AI is enabled while the button remains held.

**Trigger epoch** is a monotonically increasing runtime-only integer created
on an eligible Raw Trigger down edge. Trigger release ends the epoch. Modifier
changes do not create an epoch.

**No-replacement guarantee:** within one Trigger epoch, at most one acquisition
attempt may occur. After acquisition, every movement target must descend from
the immediately preceding unique base-box match. The first frame with zero or
more than one plausible continuation atomically publishes no target and enters
`LOST`. `LOST` publishes no target for the rest of that epoch, even if the old
box reappears alone later.

## Runtime state machine

```text
raw Trigger up / ineligible
          |
          v
        IDLE
          |
          | eligible raw Trigger down: create new epoch
          v
      ACQUIRING -- no target or tied nearest acquisition --> LOST
          |
          | one deterministic nearest base target
          v
       TRACKING -- zero/non-unique/invalid continuation --> LOST
          |                                                   |
          | exactly one unique continuation                   |
          +-------------------- TRACKING                       |
                                                              |
raw Trigger up <----------------------------------------------+
          |
          +---------------------------> IDLE
```

The acquisition attempt occurs on the first complete base inference that
observes the new epoch. If there is no accepted target, or the nearest-distance
choice is exactly tied, the epoch becomes `LOST`; it does not wait for a later
target. Detector order remains the tie break for stateless Overlay-only
selection, but it is not used to guess an acquisition tie.

While `TRACKING`, the base-frame matcher considers accepted detections of the
same target class and target area. A plausible continuation must satisfy all
of the following:

- finite, positive box geometry;
- a safe area ratio relative to the preceding confirmed base box;
- spatial continuity relative to the bounded predicted aim point;
- a distance threshold normalized through canonical 320 geometry so Center
  320 and Full Display behave consistently; and
- box overlap and displacement constraints derived from the existing
  conservative tracker policy.

Exactly one plausible candidate is required. Two plausible candidates are
ambiguous regardless of score ordering. Exact or near overlap therefore fails
closed instead of selecting the lowest numeric score. The implementation may
reuse pure geometry helpers from the legacy tracker, but it must not reuse its
recovery, hold, or replacement behavior.

The preceding confirmed **base** box is the only identity-association input for
the next frame. A refined Adaptive Zoom box never updates lock identity state.

## Component design

### Pure strict lock

`jitter_app/ai/tracking.py` retains its legacy compatibility API and gains a
separate immutable strict-lock state and observation function. Keeping the new
pure policy beside the existing geometry avoids adding a second overlapping
tracking module or changing the supported package layout.

The strict observation accepts:

- the previous immutable lock state;
- the current Trigger epoch or no epoch;
- current base detections and immutable `AimSettings`;
- sequence and capture timestamp;
- native source-frame width and height; and
- the existing output/capture viewport geometry needed by Overlay.

It returns:

- the next immutable lock state;
- a movement target or `None`;
- a complete immutable detection frame containing all accepted boxes; and
- the selected base index only while acquisition/tracking is valid.

No Tk, ONNX Runtime, DXCam, Makcu, locks, or persistence are allowed in this
pure boundary.

When there is no active Trigger epoch, the service may use existing stateless
analysis for Overlay visualization, but movement publication remains `None`.
This preserves useful real-time Overlay inspection without implying an active
movement lock.

### Trigger epoch ownership

The Tk main thread observes raw device button edges. It mirrors an immutable
runtime Trigger-lock request under the existing short AI lock; worker providers
must not read Tk variables or widgets.

On a Raw Trigger down edge, the UI decides eligibility once. For an eligible
edge it first resets AI target publication and obtains the new targeting
revision, then exposes the new epoch under the AI lock, and only then starts
normal motion. This ordering prevents the worker from claiming a newly exposed
epoch immediately before its publication is reset. An ineligible edge is
remembered as ineligible until Raw Trigger up, preventing Master, hotkey,
source, or reconnect changes from silently acquiring while the physical button
stays held.

Raw Trigger up first clears the exposed request, then resets AI targeting, and
stops normal movement through the existing cancellation path. Modifier release
stops movement but does not create a new epoch. If inference remains available
for a visible Overlay, association may continue while the Modifier is released.
If the AI generation retires during that pause, the epoch is treated as lost
and cannot reacquire after restart.

Every Test 3s run involving AI receives one unique synthetic Trigger epoch when
its production motion actually begins. It receives the same one-attempt and
irreversible-loss behavior, while Adaptive Zoom remains disabled as before.
Completion, STOP, disconnect, or test cancellation destroys the synthetic
request.

### Service ownership and cross-generation safety

`AiInferenceService.start` accepts an optional Trigger-epoch provider in
addition to the settings and Adaptive Zoom gate providers. The production UI
always supplies it. In that managed mode, a provider result of no epoch keeps
stateless Overlay visualization but publishes no movement target. Omitting the
provider entirely retains the existing stateless target-publication behavior
for isolated compatibility callers and tests; it does not opt the production
UI out of Strict Trigger Lock.

The service owns an epoch-claim record outside worker-generation-local state.
Claiming an epoch is atomic and may succeed only once. Consequently, a capture
or model restart during a held Trigger cannot let the successor worker perform
a second acquisition. A new generation observing an already claimed epoch
starts `LOST` for that epoch.

Each worker still owns its frame-to-frame strict association state. It reads
the current epoch before base inference and rechecks the epoch, targeting
revision, and generation before publishing. A release, STOP, settings reset,
or generation change during base/refinement inference discards that result.

The existing targeting revision remains the immediate stale-publication
barrier. The UI synchronously resets targeting when Confidence or Target Area
changes; an active epoch then becomes `LOST`, because either setting changes
the candidate identity set. Strength, Smoothing, Max Step, and response-curve
changes may update live because they affect movement only.

Target and selected-index publication stays atomic under the service lock. On
loss, both are cleared in the same publication. Makcu continues checking the
matching targeting revision immediately before composing/sending movement, so
an already-fetched obsolete AI target cannot leak past cancellation.

### Adaptive Zoom

The unique locked base target is the only allowed refinement seed. A base miss
or ambiguity enters `LOST` and skips refinement. A refinement result may replace
only the selected base box for that frame and never changes next-frame lock
state.

If refinement yields zero or more than one compatible candidate, the same-frame
locked 1.0x base result remains. This is a refinement fallback, not a new target
acquisition. Existing cooldown, confirmation, and `ZOOM` reporting remain
generation-local and reset whenever the normal zoom gate is false.

### Overlay and movement

All accepted detections remain available to the Overlay in detector order.

- Outside an active epoch, Overlay-only inference keeps current-frame nearest
  visualization.
- During `TRACKING`, only the unique locked box is selected and HUD lock reports
  `HEAD` or `PLAYER`.
- During `LOST`, no box is selected and HUD lock reports `NONE`; unselected
  boxes may remain visible according to runtime Overlay customization.
- The existing 150 ms Overlay freshness rule remains unchanged.

The movement snapshot is separate from visualization demand. `LOST` therefore
suppresses only AI movement. Jitter continues when selected and otherwise
eligible, combined-motion composition remains pure, and zero deltas are still
not sent to Makcu.

### Cancellation and lifecycle behavior

STOP, disconnect, shutdown, Master/hotkey disable, AI-source removal, Trigger
release, and AI errors keep their immediate cancellation semantics. They clear
movement publication before any normal interval can elapse. Re-enabling while
Raw Trigger remains held cannot create a new epoch.

Capture-mode changes, model switches, targeting-area changes, Confidence
changes, and any AI generation failure during a held epoch prevent reacquisition
for that epoch. Successful runtime rollback may restore AI availability, but AI
movement stays absent until a new Raw Trigger press. Jitter and Overlay behavior
continue according to their existing independent gates.

## Error handling

- Invalid epoch-provider values are treated as no eligible epoch and logged at
  the service boundary without crashing inference.
- Invalid geometry or non-finite matching values fail closed into `LOST`.
- Provider exceptions retain the existing AI worker error boundary: publish no
  AI target, hide Overlay, deselect AI, and preserve eligible Jitter behavior.
- An obsolete generation or targeting revision can never publish a lock result.
- No lost target is held for movement, even within the normal 150 ms freshness
  window.

## Test strategy

Implementation follows test-driven development: add each failing behavior test,
confirm the expected failure, implement minimally, then run the complete suite.

Pure tracking tests cover:

- one acquisition per epoch and deterministic nearest acquisition;
- no target and exact-distance tie on the acquisition frame;
- detector-order changes with one unique continuation;
- same-class crossings and multiple plausible continuations;
- exact/near overlap, zero matches, invalid geometry, and non-finite values;
- one-frame disappearance followed by reappearance in the same epoch;
- irreversible `LOST` until a different Trigger epoch;
- canonical behavior for Center 320 and native Full Display geometry;
- immutable complete Overlay frames and selected-index clearing; and
- Target Area and Confidence identity-signature changes.

Service tests cover:

- stale pre-Trigger snapshots are cleared before motion starts;
- release/repress epochs and atomic single-claim behavior;
- release during base or refinement inference discards publication;
- a successor AI generation cannot reclaim a held epoch;
- locked-base Adaptive Zoom and ambiguous-refinement fallback;
- Overlay-only current-frame selection versus active/lost lock rendering;
- 150 ms target freshness and measured inference status remain unchanged; and
- Jitter continues when the strict AI lock is lost.

UI and integration-style tests cover:

- epoch creation only on eligible Raw Trigger down edges;
- enabling Master/AI or reconnecting while Trigger is already held does not
  acquire;
- Modifier release/repress does not create a new epoch;
- Trigger release, STOP, disconnect, hotkey disable, source changes, and
  shutdown clear the request immediately;
- Test 3s creates and retires exactly one synthetic epoch;
- capture/model changes while held cannot reacquire after restart; and
- Tk values remain on the main thread while providers read immutable snapshots.

Default/configuration tests assert:

- exact new scalar and curve defaults;
- the Confidence boundary accepts `0.25` and rejects lower values;
- compact serialization of the new curve;
- missing/malformed values and schemas 1 through 4 use the new defaults;
- explicit schema-5 old values round-trip without migration;
- Reset Curve and startup controls show the new values; and
- schema 6 remains byte-for-byte unchanged with saving disabled.

## Documentation and repository contract

The implementation updates the active README settings table and explains the
strict no-replacement Trigger behavior. Completed historical plans remain
historical and are not rewritten.

`AGENTS.md` must be updated in the same implementation to replace only the
stateless production movement-selection paragraph with this approved Trigger
Lock decision. Outside an eligible Trigger epoch, acquisition and Overlay
visualization may still use current-frame nearest selection. No other clean-room
scope, model, capture, persistence, packaging, or dependency rule changes.

The untracked external `.onnx` files in the repository are user-owned runtime
artifacts. They are not edited, copied, deleted, packaged, or committed.

## Verification

After implementation, run the repository-required checks:

```powershell
$jitterSources = @('main.py', 'distribution_metadata.py') + @(Get-ChildItem -LiteralPath 'jitter_app' -Recurse -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
python -m py_compile @jitterSources
python -m unittest discover -s tests -v
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
```

Do not run Nuitka unless the user explicitly requests a packaged build.

Hardware verification with a connected Makcu must cover normal AI-only and
combined movement, Trigger/Modifier cycling, passing/crossing targets, loss and
new-Trigger reacquisition, both capture modes, Adaptive Zoom, Test 3s, Overlay,
model/capture switches, reconnect, STOP, hotkey disable, and shutdown.
