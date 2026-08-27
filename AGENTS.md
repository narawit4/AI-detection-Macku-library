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
implementation using only the fixed bundled model, independent motion sources,
and the constrained overlay described below; do not add training, profiles,
tray behavior, or other upstream features.

## Planned repository layout

- `main.py`: process entry point, single-instance mutex, and shutdown.
- `ui.py`: one-page Tkinter Focused Dashboard.
- `motion.py`: settings, validation, presets, and pure motion engine.
- `combined_motion.py`: pure composition of selected Jitter and AI Aim deltas.
- `ai_targeting.py`: immutable AI settings, target selection, and movement.
- `ai_detection.py`: fixed-contract ONNX Runtime detector.
- `ai_capture.py`: centered DXCam capture wrapper.
- `ai_service.py`: generation-safe capture and inference worker.
- `overlay.py`: centered, click-through, capture-excluded detection view.
- `distribution_metadata.py`: validates, reviews, and executes the canonical
  packaging command and release-material plan.
- `nuitka-package.config.yml`: explicitly bundles ONNX Runtime's DirectML DLL.
- `makcu_service.py`: Makcu connection, callbacks, movement, and cleanup.
- `hotkeys.py`: Windows global-hotkey polling.
- `settings.py`: independent schema-aware atomic configuration.
- `models/all_games_320.onnx`: approved fixed AI Aim model resource.
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
- AI Aim uses only the fixed centered 320-by-320 capture and bundled model,
  prefers heads over players, and shares the same Trigger/Modifier gate.
- Combined movement sums current source deltas; Jitter continues when AI Aim
  has no target.
- The optional overlay starts off and is independent of source selection. It
  is a centered 320-by-320 configurable-color detection view that must be
  click-through and excluded from capture. Head-box visibility affects only
  the overlay; AI Aim keeps its head-first targeting behavior.
- STOP immediately cancels movement, hides the overlay, and ends its inference
  demand. Disable, disconnect, and source changes immediately cancel movement;
  AI inference continues only while the visible independent overlay requires
  it. Shutdown ends both movement and inference.
- An AI runtime error hides the overlay and deselects AI Aim. With Jitter still
  selected under Master, continue or restart Jitter through the same gate;
  an AI-only failure disarms Master.
- `Test 3s` uses the production engine for the sources selected at test start,
  bypasses Trigger temporarily, requires Makcu, and remains immediately
  interruptible by STOP or disconnect.
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
- Schema 5 persists validated settings, including overlay color and head-box
  visibility; do not persist motion-source selection, Master state, overlay
  visibility, AI targets, snapshots, FPS, provider, or runtime status.
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
- Do not add alternate AI models, training, profiles, tray, Pillow,
  Pystray, Torch, Ultralytics, OpenCV, or other unapproved ML dependencies
  without an explicit new design decision.

## Verification

After implementation changes, run:

```powershell
python -m py_compile main.py ui.py motion.py combined_motion.py ai_targeting.py ai_detection.py ai_capture.py ai_service.py overlay.py makcu_service.py hotkeys.py settings.py sound_service.py liquid_widgets.py distribution_metadata.py
python -m unittest discover -s tests -v
python -c "import makcu, serial, onnxruntime, dxcam, comtypes, numpy"
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
