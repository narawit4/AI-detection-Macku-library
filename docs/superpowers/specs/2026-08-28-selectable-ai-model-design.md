# Selectable AI Model Design

**Date:** 2026-08-28

**Status:** Approved

## Summary

Jitter will let the user browse to an external ONNX model for the current
application run. The bundled `models/all_games_320.onnx` remains the default,
the trusted fallback, the packaged model, and the model checked by
`--ai-runtime-self-check`.

This is an explicit, narrow design decision that amends the previous
fixed-model-only scope. It does not permit alternate runtimes, model training,
downloads, profiles, or model redistribution. A custom model must satisfy the
existing fixed detector contract and uses the existing ONNX Runtime
DirectML/CPU provider policy.

## Goals

- Add `Browse...` and `Use Default` controls for selecting an ONNX model.
- Keep model selection runtime-only; every launch starts with the bundled
  default model.
- Validate a candidate off the Tk thread before committing the selection.
- Stop active AI work during a switch and resume current demand after success.
- Roll back to the previously active model when validation or startup fails.
- Preserve STOP, source selection, Master state, Overlay demand, conservative
  tracking, target-area selection, adaptive zoom, and generation safety.
- Keep schema 5, packaging inputs, dependency pins, and the canonical bundled
  model self-check unchanged.

## Non-goals

- Persisting a model path, name, hash, or recent-model list.
- Copying external models into `models/` or any application-owned cache.
- Scanning directories or maintaining a model catalog.
- Downloading, training, converting, editing, or profiling models.
- Supporting Torch, Ultralytics, OpenCV, alternate ONNX runtimes, or a second
  inference contract.
- Bundling or licensing a user-selected external model in a Jitter release.
- Changing class meanings, capture size, preprocessing, or detector output
  parsing.

## Model Contract

The existing `OnnxDetector` contract remains authoritative:

- one float input named `images` with shape `[1, 3, 320, 320]`;
- one float output named `output0` with shape `[1, 300, 6]`;
- the existing RGB normalization and fixed 320-by-320 centered capture;
- detection rows interpreted as `x1, y1, x2, y2, confidence, class_id`;
- class `0` interpreted as player and class `7` interpreted as head;
- unknown class IDs ignored by current targeting behavior;
- DirectML requested first with the existing CPU fallback behavior.

The structural contract can be validated, but the program cannot prove that a
model author assigned the intended semantic meaning to class IDs 0 and 7. The
user is responsible for choosing a model with those class semantics.

## Runtime Model State

Introduce an immutable runtime model choice containing:

- a resolved absolute `Path`;
- a display name derived from the filename;
- whether it is the bundled default.

The state is owned by `JitterApp` and never enters `AppConfig`. At startup the
active choice is always `model_resource_path()`, even if an unsupported
`model_path` key is present in configuration data.

The UI also owns a model-switch record containing the candidate choice, the
previous choice, a monotonically increasing switch token, and the switch
phase. It deliberately does not cache AI demand: every restart decision reads
the current lifecycle state. The switch token is separate from inference and
motion generations. It prevents validation or runtime events from an obsolete
switch from committing state.

## Candidate Selection and Validation

`Browse...` opens a native file dialog restricted to `.onnx` files. Basic path
validation occurs on the Tk thread and is limited to cheap filesystem checks:

- a nonempty selection;
- case-insensitive `.onnx` suffix;
- an existing regular file;
- successful absolute-path resolution.

Canceling the dialog changes nothing. A basic validation failure leaves the
current AI runtime untouched and shows a concise footer error.

Full model loading and contract validation must never occur on the Tk thread.
A one-shot, daemon model-validation worker constructs `OnnxDetector` for the
candidate without starting DXCam, capture, or inference. It reports success or
a sanitized failure through the existing Tk queue boundary. Detailed exception
information goes to `app.log`.

The validation worker has its own stop event and switch token. Results from an
obsolete token are ignored. The worker retains no model session after its
result; a normal AI generation constructs its own detector so inference state
has one clear owner.

## Generation-safe Switch Flow

When a valid path is chosen:

1. Record the candidate, previous model, and a new switch token.
2. Disable `Browse...` and `Use Default`, but keep STOP and all emergency paths
   available.
3. Immediately cancel normal mouse movement that consumes AI targets and
   invalidate the current target, tracker, zoom confirmation, selected Overlay
   highlight, and queued zoom status.
4. If AI inference is active, stop its generation through the existing
   generation-safe service path. Do not change Master, source selection, or
   Overlay visibility merely because a model is switching.
5. Run one-shot model validation off the Tk thread.
6. On validation success, make the candidate the active runtime choice and,
   only if current lifecycle state still requires AI, start a fresh inference
   generation with the candidate path.
7. Commit the switch after the fresh generation emits `ready`. If there is no
   current AI demand, commit immediately after validation and leave inference
   stopped.
8. Re-enable the model controls and report `Using model: <filename>`.

`Use Default` follows the same flow. It is disabled when the default is already
active. The bundled model still goes through generation-safe switching, but its
known path does not need a separate user-facing file check.

Each inference generation receives an immutable model-path snapshot. An old
worker must never consult a later model choice. Existing AI event epochs and
service generations reject stale `ready`, `error`, FPS, target, detection, and
zoom publication.

## Rollback and Cancellation

If candidate validation fails, restore the previous model choice. If AI is
still required by the current lifecycle, start a fresh generation using that
previous path. Report a concise message such as
`Model rejected; restored <filename>`.

If candidate validation succeeds but the normal candidate generation fails to
reach `ready`, perform the same rollback once. A rollback generation is marked
explicitly so its `error` cannot recursively request another rollback. If the
rollback model also fails, exit the switch state and invoke the existing AI
runtime error behavior: hide Overlay, deselect AI Aim, preserve eligible Jitter
movement, or disarm Master for AI-only demand.

Rollback evaluates current demand at completion; it never relies on demand
state observed when the dialog closed. If the user removes AI demand while
switching, the program restores the previous model selection without
restarting inference.

STOP, shutdown, disconnect, removal of the AI source, or removal of final
Overlay demand invalidates the switch token before stopping services. Late
validation and inference events are ignored and cannot auto-restart AI. STOP
continues to hide Overlay and cancel movement immediately.

Model switching is unavailable during every `Test 3s` pending, loading, and
running state. The user must let the test complete or press STOP first.

## UI Design

Add a `MODEL` row to `AI AIM SETTINGS` inside the existing scrollable Motion
page. It displays one of:

- `Default · all_games_320.onnx`
- `Custom · <filename>`

The row provides `Browse...` and `Use Default`. Only the filename is rendered;
absolute paths are limited to diagnostics. During validation or restart, the
row shows `Loading · <filename>` and both model controls are disabled. The
outer window remains fixed at 840 by 620 and the persistent red STOP button
remains visible.

The footer uses short actionable messages. Existing AI status, provider, FPS,
and ZOOM metrics continue to describe the active inference generation. ZOOM is
reset to 1.0x as soon as a switch begins.

## Configuration and User Data

`SCHEMA_VERSION` remains 5. Model selection is runtime-only and must not appear
in `config.json`, `config.json.bak`, presets, or any other user-data file.
Changing the model must not schedule a configuration save. Opening a new
process always selects the bundled default.

The existing rule that response curve is the only newly persisted AI setting
remains intact. Future unsupported schemas remain byte-for-byte protected.

## Packaging and Self-check

The canonical Nuitka command continues to include the existing `models/`
directory, whose approved release content remains
`models/all_games_320.onnx`. External models are referenced in place and are
never copied into build output or release materials.

`--ai-runtime-self-check` continues to verify the bundled model's fixed
SHA-256, fixed detector contract, and `DmlExecutionProvider`. A custom model is
not allowed to redefine release readiness or replace the trusted fallback.

No dependency, license manifest, package configuration, or release-material
change is required.

## Threading and Logging

- Tk variables, file dialogs, widget state, and footer text remain on the main
  thread.
- Model construction and contract validation run on a daemon worker.
- Worker results enter Tk only through the existing queue/`after` boundary.
- Switch, inference, motion, and targeting generations remain independently
  invalidatable.
- Locks protect short immutable-state swaps only; no model load, filesystem
  dialog, capture operation, or stop wait occurs while holding a shared lock.
- Detailed paths and exceptions go to thread-safe logging. UI errors reveal
  only the filename and an actionable summary.

## Testing

Development follows TDD. Required hardware-free coverage includes:

- candidate path validation, cancel behavior, suffix handling, missing files,
  and resolved paths;
- fixed input/output contract acceptance and rejection;
- one-shot validation without DXCam creation or inference;
- successful active and idle switches;
- rollback after validation failure and after candidate runtime failure;
- rollback failure reaching the existing fail-closed AI error path exactly
  once;
- rapid repeated selections and stale validation/AI generation events;
- STOP, shutdown, disconnect, source removal, and Overlay-demand removal during
  every switch phase;
- current demand, rather than demand observed at selection time, controlling
  restart;
- model controls disabled during switch and `Test 3s`;
- fixed window geometry, scrolling, theme readability, and STOP visibility;
- no model data in schema 5 saves, backups, or reloads;
- startup always restoring the bundled default;
- packaging metadata and self-check continuing to reference only the bundled
  model.

After implementation, run the complete repository verification sequence from
`AGENTS.md`. Do not run Nuitka unless the user explicitly requests a packaged
build. Hardware acceptance should additionally exercise a valid compatible
custom model, an invalid-contract ONNX file, switching while AI movement is
active, rollback, STOP during loading, reconnect, Overlay-only demand, and all
Jitter/AI source combinations.

## Acceptance Criteria

- The user can browse to a compatible external ONNX model and use it for the
  current process.
- Active AI work stops safely during a switch and resumes only when current
  demand permits.
- An invalid or failed candidate restores the previous model automatically.
- A failed rollback enters the existing fail-closed AI error path without a
  retry loop.
- Stale model, inference, target, and zoom events cannot publish or restart
  work.
- STOP and shutdown remain immediate at every switch phase.
- Model choice is never persisted, copied, packaged, or used by the canonical
  bundled-model self-check.
- The default application behavior remains the bundled approved model.
