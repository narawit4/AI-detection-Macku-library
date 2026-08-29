# AGENTS.md

This file gives Codex repository-specific guidance for the standalone Jitter
Windows application.

## Project overview

Jitter is a Windows-only Python/Tkinter desktop application for a Makcu USB
device. It connects automatically and produces configurable smooth
two-dimensional mouse Jitter while the selected Trigger and optional Modifier
are held.

This repository is independent from EverFall Jitter. Do not import its source
or read its configuration. The approved AI Aim scope is a clean-room
implementation using the bundled startup-default model, independent motion
sources, and the constrained overlay described below. Only runtime browsing of
contract-compatible external `.onnx` files is allowed; do not add training,
profiles, downloads, copying, persistence, tray behavior, or other upstream
features.

## Supported repository layout

- `main.py`: process entry point, single-instance mutex, and shutdown.
- `distribution_metadata.py`: validates, reviews, and executes the canonical
  packaging command and release-material plan.
- `jitter_app/__init__.py`: application package marker.
- `jitter_app/resources.py`: bundle and repository resource resolution.
- `jitter_app/ai/__init__.py`: AI package marker.
- `jitter_app/ai/capture.py`: centered DXCam capture wrapper.
- `jitter_app/ai/detection.py`: dual-contract ONNX Runtime detector boundary for
  legacy post-NMS `[1,300,6]` and raw single-class `[1,5,K]` outputs.
- `jitter_app/ai/model_selection.py`: runtime-only external ONNX selection and
  validation.
- `jitter_app/ai/service.py`: generation-safe capture and inference worker.
- `jitter_app/ai/targeting.py`: immutable AI settings, target selection, and
  movement.
- `jitter_app/ai/tracking.py`: legacy pure tracker retained for compatibility
  tests; production target selection is stateless and current-frame only.
- `jitter_app/ai/resize.py`: pure resizing and coordinate mapping for supported
  model sizes.
- `jitter_app/ai/yolo.py`: pure NumPy decoder for the raw single-class
  Ultralytics contract.
- `jitter_app/ai/zoom.py`: pure adaptive zoom geometry and same-frame refinement
  composition.
- `jitter_app/motion/__init__.py`: motion package marker.
- `jitter_app/motion/engine.py`: settings, validation, presets, and pure motion
  engine.
- `jitter_app/motion/combined.py`: pure composition of selected Jitter and AI
  Aim deltas.
- `jitter_app/device/__init__.py`: device package marker.
- `jitter_app/device/makcu.py`: Makcu connection, callbacks, movement, and
  cleanup.
- `jitter_app/device/hotkeys.py`: Windows global-hotkey polling.
- `jitter_app/device/display_timing.py`: primary-display cadence detection and
  pure runtime policy.
- `jitter_app/presentation/__init__.py`: presentation package marker.
- `jitter_app/presentation/ui.py`: one-page Tkinter Focused Dashboard.
- `jitter_app/presentation/widgets.py`: shared Tk widgets and styles.
- `jitter_app/presentation/overlay.py`: centered, click-through, capture-excluded
  detection and AI runtime status view.
- `jitter_app/presentation/sound.py`: sound service.
- `jitter_app/config/__init__.py`: configuration package marker.
- `jitter_app/config/store.py`: independent schema-aware atomic configuration.
- `nuitka-package.config.yml`: explicitly bundles ONNX Runtime's DirectML DLL.
- `models/all_games_320.onnx`: approved bundled startup-default AI Aim model
  resource; this is the only bundled model.
- `licenses/`: exact dependency notices, provenance manifest, and required
  GPL/LGPL source archives.
- `tests/`: hardware-free unit and integration-style tests.
- `run_gui.bat`: source launcher.
- `gen.bat`: explicit on-demand Nuitka packaging.
- `docs/superpowers/`: design specifications and implementation plans.

## Running and dependencies

Run from the repository root:

```powershell
python main.py
```

or double-click `run_gui.bat` after it is implemented.

Runtime dependencies belong in `requirements.txt` and stay exactly pinned.
The approved AI stack is ONNX Runtime DirectML, DXCam, and NumPy; do not add
Torch, Ultralytics, OpenCV, or alternate model runtimes. Tkinter is included
with the supported Windows Python installation.

## Architecture and threading

- Keep Tk widget and Tk-variable access on the main thread.
- Device callbacks and workers must marshal UI work through a queue or
  `after(0, ...)`.
- Keep the motion engine pure and independent of Tkinter and Makcu.
- Keep combined-motion composition pure and independent of Tkinter and Makcu.
- Keep adaptive zoom geometry and refinement composition pure and independent
  of Tkinter and Makcu.
- Keep target selection pure and independent of Tkinter and Makcu.
- Keep display-cadence policy pure; display detection stays at the Windows
  boundary and its result is runtime-only.
- Share immutable motion snapshots with the mover under a short lock.
- Use daemon workers and explicit stop events/generation identifiers for
  connection, movement, Test Run, and hotkey polling.
- Ignore stale results from obsolete connection or movement generations.
- Keep blocking Makcu calls off the Tk event loop.
- Centralize connection-state transitions and emergency-stop behavior.
- Never let STOP, disconnect, hotkey disable, or shutdown wait for a normal
  movement interval before signalling cancellation.

## Scope and behavior

- The interface is one fixed-size English page using the approved Focused
  Dashboard layout.
- Keep the red STOP button visible when Advanced Settings expands.
- Jitter and AI Aim are independent selected sources; both start unselected.
  Master and the global hotkey arm the selected sources, while actual movement
  requires Trigger plus the configured Modifier, if any.
- AI Aim uses the fixed centered 320-by-320 capture and bundled startup-default
  model, considers accepted heads and players together, and shares the same
  Trigger/Modifier gate.
  `Browse...` accepts only runtime external ONNX models whose `images` float input
  is exactly `[1,3,N,N]` for N in 160, 320, or 640 and whose `output0` float output
  is exactly either legacy `[1,300,6]` or raw single-class `[1,5,K]`, with
  `(N,K)` exactly `(160,525)`, `(320,2100)`, or `(640,8400)` and safe
  `task=detect`/one-class `names` metadata. Custom metadata-map keys/values are
  strings; additional all-string Ultralytics fields are allowed. The string-valued
  `names` field is safely parsed with `ast.literal_eval` and must equal exactly
  `{0: "<non-empty label>"}`. Capture, Overlay, targeting, movement,
  and Adaptive Zoom
  remain in canonical 320-by-320 coordinates; detector output is scaled back
  before publication. Validate off the UI thread, pause AI during a switch, and
  after its exact ready event restart the eligible AI runtime and motion. A
  candidate startup failure makes exactly one automatic rollback attempt to
  restart the previous model. Legacy `[1,300,6]` behavior, downstream canonical
  `Detection`, and the bundled startup model remain unchanged. Never download,
  copy, package, or persist an external model or its path; every launch starts
  with the bundled 320 model.
- On every base frame, filter detections by confidence and supported class,
  derive the configured aim point for every accepted head and player, and
  select the point with the shortest Euclidean distance to the centered
  crosshair at `(160, 160)`. Preserve detector order as the exact-distance tie
  break. Do not use prior identity, ambiguity holds, recovery confirmation, or
  replacement delays; publish the current-frame selection immediately.
- Derive runtime cadence from the primary display: cap capture at 240 FPS and
  run the servo at twice display refresh, clamped to 120-480 Hz. Fall back to
  120 FPS capture and a 240 Hz servo when detection is unavailable or invalid.
  Display, servo, and measured inference cadence are runtime status only.
- AI Aim uses a five-point distance-to-speed response curve at 0%, 25%, 50%,
  75%, and 100% distance. The first point is fixed at zero; the other four are
  adjustable exact ordered percentages. Curve output is scaled by Strength,
  approached using time-based Smoothing, and bounded by Max Step. Reset Curve
  restores the complete default.
- Consume each fresh AI target through time-based servo microsteps and discard
  any unconsumed target after 150 ms. This does not change Jitter composition
  or immediate cancellation behavior.
- Adaptive Zoom is automatic and has no persisted control. Every frame first
  performs full-field 1.0× target acquisition; only an already-selected small
  target may receive a same-frame 1.5× or 2.0× refinement pass. That second
  pass runs only during connected, Master-armed, AI-selected normal movement
  with the configured Trigger and Modifier active. It is excluded while idle,
  for Overlay-only inference, and during `Test 3s`.
- `ZOOM` is runtime status only and reports 1.0×, 1.5×, or 2.0×. If refinement
  is ineligible or cannot produce a compatible result, the same-frame 1.0×
  base result remains. A successful refinement replaces only the selected base
  box; its box is mapped back to the original frame for Overlay rendering and
  unrelated base boxes remain. Adaptive Zoom does not magnify the display or
  recover targets the base pass never detected.
- Recoil-stable zoom confirmation is separate from movement publication. It
  observes the current nearest base target and may limit refinement to 1.5x,
  but it must not withhold that base target from AI movement while confirming
  2.0x refinement eligibility.
- A requested 2.0x refinement is capped at 1.5x until confirmation and a fixed
  100 ms cooldown both pass. A normal refinement miss resets confirmation and
  extends cooldown without adding an inference call or holding a stale target.
- Zoom stability is local to one AI generation and resets when the movement
  zoom gate is false. Base target selection remains stateless across frames.
- Combined movement sums current source deltas; Jitter continues when AI Aim
  has no target.
- The optional overlay starts off and is independent of source selection. It
  is a centered 320-by-320 configurable-color detection view that must be
  click-through and excluded from capture. Its top-left runtime HUD reports
  FPS, provider, zoom, and the current-frame `HEAD`, `PLAYER`, or `NONE` lock;
  a detection frame older than 150 ms reports `NONE`. Head-box visibility
  affects only the overlay; AI Aim still considers hidden head boxes for
  nearest-target selection.
- STOP immediately cancels movement, hides the overlay, and ends its inference
  demand. Disable, disconnect, and source changes immediately cancel movement;
  AI inference continues only while the visible independent overlay requires
  it. Shutdown ends both movement and inference.
- An AI runtime error hides the overlay and deselects AI Aim. With Jitter still
  selected under Master, continue or restart Jitter through the same gate;
  an AI-only failure disarms Master.
- `Test 3s` uses the production engine for the sources selected at test start,
  bypasses Trigger temporarily, requires Makcu, and remains immediately
  interruptible by STOP or disconnect. Model changes are unavailable during a
  Test 3s run.
- The configurable global hotkey defaults to `-` and toggles once per press.
- Makcu connection uses the supported `makcu` library with automatic
  reconnection and button monitoring.
- Makcu movement is two-dimensional. Do not treat the wheel as a Z axis.
- Preserve clamping, acceleration limits, fractional accumulation, and the rule
  that excess movement is discarded instead of queued.
- Closing the window exits the app; there is no system tray.

## UI conventions

- Reuse shared palette, spacing, font, and ttk style constants.
- Primary cyan controls use dark readable text; secondary controls retain light
  text in all states.
- Connection colors are green, amber, and red for connected, connecting, and
  disconnected.
- Advanced controls scroll inside the existing window rather than changing its
  outer dimensions.
- Numeric controls provide both a slider and an exact-value input.
- Show concise actionable errors in the footer; detailed diagnostics go only
  to `app.log` through Python's thread-safe logging facilities.

## Configuration and user data

- `config.json`, `config.json.bak`, and `app.log` are local user data and must
  remain ignored by Git.
- Never copy, reset, or overwrite EverFall Jitter user data.
- Validate loaded values and preserve safe defaults for malformed data.
- Never overwrite a config with a newer unsupported schema; run with safe
  in-memory defaults and leave that file untouched.
- Schema 5 persists validated settings, including overlay color, head-box
  visibility, and the optional AI response curve. A missing or malformed curve
  uses the complete default, and schemas 1-4 also use that curve default.
  Schema 6 is an unsupported future schema: load safe in-memory defaults,
  disable saving, and leave its source file byte-for-byte unchanged.
- The response curve is the only new persisted setting. Do not persist
  motion-source selection, Master state, overlay visibility, target history,
  AI targets, model selection or external model paths, snapshots, FPS, provider,
  cadence, zoom status, or other runtime state. Adaptive Zoom and adaptive
  cadence add no persisted control.
- Write configuration through a temporary file, flush and `fsync`, keep a
  backup, and replace atomically.
- Do not persist held-button or Moving state.

## Development workflow

- Use test-driven development for features and fixes: add a failing test, verify
  the expected failure, implement minimally, then run the complete suite.
- Preserve unrelated user changes and generated artifacts.
- Do not edit `build-output/`, `dist/`, `*.build/`, `*.dist/`, `__pycache__/`, or
  `app.log` as source.
- Do not run Nuitka after ordinary feature changes. Package only when the user
  explicitly requests it or asks to run `.\gen.bat` or
  `python .\distribution_metadata.py --build`.
- Keep `.\gen.bat` a no-argument, exact-confirmation wrapper. Never expand or
  forward batch arguments; use the Python launcher for help, review, build
  automation, and any constructed argument vector because `cmd.exe` parses
  metacharacters and redirects before batch code runs.
- A binary release must include `LICENSE`, `THIRD_PARTY_NOTICES.md`, and the
  complete `licenses/` directory beside the executable. Follow
  `licenses/README.md`; Jitter source alone does not satisfy every dependency.
- The canonical Nuitka plan must load `nuitka-package.config.yml`, then run
  `build-output\Jitter.exe --ai-runtime-self-check` successfully with
  `DmlExecutionProvider` before copying release materials or reporting success.
- Do not add alternate bundled or packaged AI models, training, profiles,
  downloads, copying, tray, Pillow, Pystray, Torch, Ultralytics, OpenCV, or
  other unapproved ML dependencies without an explicit new design decision.

## Verification

After implementation changes, run:

```powershell
$jitterSources = @('main.py', 'distribution_metadata.py') + @(Get-ChildItem -LiteralPath 'jitter_app' -Recurse -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
python -m py_compile @jitterSources
python -m unittest discover -s tests -v
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
```

Hardware-dependent changes additionally require a connected Makcu device to
verify connection, Trigger/Modifier buttons, each Jitter/AI Aim source
combination, combined movement, reconnect, Test Run, global hotkey, STOP,
shutdown, and the optional click-through capture-excluded overlay.

For an explicitly requested confirmed packaged build, run `.\gen.bat` and
type `BUILD`, then verify
`build-output\Jitter.exe` separately.
