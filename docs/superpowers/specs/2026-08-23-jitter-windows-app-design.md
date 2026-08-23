# Jitter Windows App Design

Date: 2026-08-23
Status: Approved in chat on 2026-08-23

## Overview

Create a new, independent Windows-only Python application in
`C:\Users\User\Desktop\Jitter`. The app preserves the proven smooth Jitter
motion behavior and Makcu connection protocol from EverFall Jitter, but it is a
clean rewrite with a one-page English Tkinter interface and no AI or training
features.

The new project owns its source, Git history, configuration, logs, executable,
and tests. It must never read or modify EverFall Jitter user data.

## Goals

- Provide reliable configurable two-dimensional Jitter through a Makcu device.
- Connect to Makcu automatically and recover cleanly from disconnections.
- Gate movement behind an explicit enabled state and the selected
  Trigger/Modifier combination.
- Keep a configurable global enable/disable hotkey, defaulting to `-`.
- Present all version-one controls in one simple, fixed-size English window.
- Preserve safe motion behavior: smoothing, ramping, acceleration limits,
  per-report clamping, and fractional accumulation without a movement backlog.
- Keep the code modular and independently testable.
- Package only on explicit request through `gen.bat`.

## Non-goals

Version one will not include:

- AI detection, ONNX Runtime, model management, tracking, or training.
- Profiles or profile hotkeys.
- Center-screen overlays.
- System-tray behavior.
- An in-app log viewer.
- DPI normalization, Motion Analyzer, profile import/export, or comparison.
- Reading, migrating, or overwriting EverFall Jitter's `config.json`.

The app may append technical diagnostics to its own `app.log`; the file is not
shown inside the UI.

## Technology and project structure

The app uses Python 3.10 or newer, standard Tkinter/ttk, and the existing
`makcu` Python package. Runtime code is separated by responsibility:

- `main.py`: process entry point, Windows single-instance mutex, and lifecycle.
- `ui.py`: the one-page Tkinter window and all Tk-variable/widget access.
- `motion.py`: immutable motion settings, validation, presets, and the pure
  smooth Jitter engine.
- `makcu_service.py`: background connection setup, automatic reconnection,
  Makcu button callbacks, movement reports, and disconnect cleanup.
- `hotkeys.py`: edge-triggered Windows global-hotkey polling and cancellation.
- `settings.py`: schema-aware independent configuration and atomic persistence.
- `tests/`: unit and integration-style tests that do not require real hardware.
- `AGENTS.md`: repository-level Codex guidance generated as the project's
  `/init` setup and kept current when architecture or verification changes.

No module in this project imports EverFall source files. Reused behavior is
rewritten into the new modules and covered by new tests.

## UI design

The selected layout is **A — Focused Dashboard**. The English-only window is
approximately 720 by 680 pixels. It remains the same outer size when Advanced
Settings opens; the content area scrolls internally when necessary.

The visual style is restrained dark graphite with cyan primary actions, green
connected state, amber connecting state, and red disconnected/emergency state.
Secondary actions keep light text. The red `STOP` button remains visible and is
never placed inside the collapsible Advanced area.

The page contains:

1. A header with the app name and `Connected`, `Connecting`, or `Disconnected`.
2. A Device card with concise Makcu details and `Reconnect`.
3. A Main Control card with a large `Enable Jitter` toggle and a runtime state
   of `Disabled`, `Armed`, or `Moving`.
4. A Trigger card with Trigger, optional Modifier, and configurable global
   hotkey.
5. An Action card with preset selection, `Test 3s`, and `STOP`.
6. Quick Jitter controls for Angle, Strength, Horizontal Jitter, Vertical
   Jitter, and Jitter Rate. Each value has a slider and exact numeric input.
7. A collapsible Advanced Settings section for Waveform, Randomness, Axis
   Phase, Smoothness, Ramp, Motion Curve, Update Rate, Acceleration,
   Deceleration, and Max Step.
8. A compact footer for the latest actionable status or error.

The six initial presets are Ultra Stable, Soft, Balanced, Fast Response,
Strong Shake, and Extreme, using the proven numeric defaults from EverFall.
Changing a control updates the in-memory settings immediately and schedules a
debounced configuration save.

Closing the window exits the process; it does not minimize to a tray.

## Motion model

`MotionSettings` is an immutable value object. The engine accepts settings,
delta time, elapsed run time, and a random source, and returns one integer
Makcu report `(x, y)` per step.

The engine provides:

- Screen-coordinate direction: 0 degrees right and 90 degrees down.
- A base strength vector in pixels per second.
- Independently configurable horizontal and vertical Jitter.
- Sine, Triangle, Square, and Random blend waveforms.
- Configurable Jitter frequency, axis phase, and randomness.
- Linear, Ease-in, and S-curve ramping.
- Time-based smoothing and acceleration/deceleration limiting.
- Fractional residual accumulation before integer reports.
- A per-axis max-step clamp that discards excess instead of queueing it.

The version-one validation ranges are:

- Angle: 0-360 degrees.
- Strength and horizontal/vertical Jitter: 0-500 pixels per second.
- Smoothness: 1-100 percent.
- Update rate: 20-500 Hz.
- Ramp: 0-2000 milliseconds.
- Jitter rate: 0.1-60 Hz.
- Randomness: 0-100 percent.
- Axis phase: 0-360 degrees.
- Max step: 1-50 pixels per report.
- Acceleration and deceleration: 1-10000 pixels per second squared.

Invalid typed values are rejected in the UI without replacing the last valid
motion snapshot. Loading invalid persisted values clamps or restores them to a
safe default.

## Runtime behavior and data flow

Startup performs these operations in order:

1. Acquire a new-project Windows named mutex to prevent a second instance.
2. Create the UI and load this project's settings.
3. Start global-hotkey polling.
4. Start Makcu connection on a daemon worker thread.

Makcu connection uses `create_controller(debug=False, auto_reconnect=True)`,
registers a connection-change callback, enables button monitoring, and
registers a button callback. Controller callbacks never touch Tk widgets;
events are forwarded to the Tk thread through a queue or `after(0, ...)`.

Enabling Jitter enters `Armed`. Movement starts only while the configured
Trigger is held and, when configured, the Modifier is also held. Releasing
either required button stops movement. A single motion worker owns its stop
event and generation identifier so an obsolete worker cannot resume.

The motion worker reads an immutable settings snapshot under a short lock,
steps the motion engine, sends non-zero reports through `makcu.move(x, y)`, and
uses the stop event for interruptible timing. No Tk calls occur on this worker.

`Test 3s` uses the same worker and engine but temporarily bypasses the trigger
requirement. It requires a connected Makcu device, can be interrupted by
`STOP`, and restores the normal Armed/Disabled state when finished.

The default global hotkey is keyboard key `-`. A capture control lets the user
replace it with another Windows virtual key; Escape cancels capture. Polling is
edge-triggered so holding the hotkey produces one toggle rather than repeated
toggles.

## State and safety rules

- `STOP` always disables Jitter, clears held-button state, invalidates the
  active motion generation, and signals the worker stop event immediately.
- A Makcu disconnect performs the same emergency-stop operation before showing
  `Disconnected`.
- Manual reconnect stops movement, disconnects the exact old controller on a
  worker, and begins a new generation-safe connection attempt.
- Connection errors leave the UI responsive and expose a short footer message.
- A movement exception stops the worker and reports the exception type in the
  local log.
- Window shutdown stops movement and hotkey polling before disconnecting the
  controller. Slow controller cleanup remains off the Tk thread.
- Any callback or worker result from an outdated generation is ignored.
- The red `STOP` control is enabled whenever it can make the state safer.

## Configuration and diagnostics

The new project writes `config.json` beside the source script or beside the
packaged executable. It contains a schema version, all motion fields, Trigger,
Modifier, global-hotkey virtual key/name, and the selected preset name. It
does not persist a held-button or Moving state, and Jitter starts Disabled on
every process launch.

Writes use a temporary file, flush and `fsync`, preserve the previous document
as `config.json.bak`, and atomically replace the active file. A corrupt document
falls back to safe defaults and produces a diagnostic entry. A newer unsupported
schema is left untouched; the session uses safe in-memory defaults and skips
automatic saves so it cannot destroy data from a newer app version.

`app.log` receives timestamped lifecycle, connection, safety-stop, and failure
messages through Python's thread-safe logging facilities. There is no Tk log
queue or in-app log widget.

## Error handling

Expected user-correctable failures use concise English footer messages:

- Makcu not found or disconnected.
- Invalid numeric setting.
- Hotkey capture cancelled or unavailable.
- Test Run requested without a connection.
- Configuration load/save failure.

Unexpected exceptions are logged with their type and message. They do not
trigger an automatic retry loop outside Makcu's supported auto-reconnect
behavior. Manual `Reconnect` remains available.

## Testing and verification

Development follows test-first changes. Automated tests cover:

- Direction, strength, waveforms, independent axes, balanced zero-centered
  Jitter, ramp curves, smoothing, acceleration limits, max-step behavior, and
  fractional accumulation.
- Settings validation, defaults, schema handling, atomic replacement, and
  backup preservation.
- Trigger/Modifier transitions, hotkey edge detection, STOP, Test Run timeout,
  and stale-generation rejection.
- Makcu service success, expected connection failure, unexpected exception,
  disconnect, auto-reconnect signal, and cleanup through a fake controller.
- Source guards confirming worker callbacks marshal UI changes to the Tk
  thread.

Standard verification is:

```powershell
python -m py_compile main.py ui.py motion.py makcu_service.py hotkeys.py settings.py
python -m unittest discover -s tests -v
python -c "import makcu"
```

Hardware verification separately confirms automatic connection, every Makcu
button option, disconnect/reconnect behavior, live movement, hotkey toggling,
Test Run, and STOP on a connected device.

## Running and packaging

`run_gui.bat` launches `python main.py` from the project directory and preserves
an error console when startup fails.

`gen.bat` is the only packaging entry point. It installs declared dependencies,
runs syntax/tests/import verification, and creates a Nuitka one-file Windows GUI
executable at `build-output\Jitter.exe`. Normal source development does not run
Nuitka, and generated build output is excluded from Git.

Runtime dependencies are intentionally limited to the pinned `makcu` package
and Python's standard library. Build-only dependencies are installed by
`gen.bat`.

## Acceptance criteria

The first version is complete when:

1. `python main.py` opens the approved one-page English Focused Dashboard.
2. The source tree contains no AI, training, profile, overlay, or tray runtime.
3. The UI remains responsive during connection, reconnect, movement, Test Run,
   and shutdown.
4. A connected Makcu produces validated smooth Jitter only under the approved
   Enable and Trigger/Modifier rules.
5. Global hotkey, STOP, disconnect, and close all halt movement safely.
6. Advanced Settings never expand the outer window and never hide STOP.
7. Configuration is independent, atomic, backward-safe within this new schema,
   and never imports EverFall user data.
8. All automated verification commands pass.
9. `gen.bat` produces `build-output\Jitter.exe` only when explicitly run.
10. Root `AGENTS.md` accurately documents project scope, architecture, safety,
    verification, user-data handling, and the no-routine-build policy.
