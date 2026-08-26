# Jitter

Jitter is a small Windows-only Tkinter controller for a Makcu USB device. It
has a fixed-size English Liquid Split Console with Control and Motion pages.
It sends vertical paired pulses: one equal pair moves up then down, and the
following equal pair moves down then up; this two-pair order repeats. Complete
pulse pairs have zero intended displacement; results vary with the receiving
application's input processing. Choose a Trigger and optional Modifier, arm or
stop movement, run a three-second test, and assign a global toggle hotkey.

## Jitter motion controls

In Jitter mode, the Motion page provides these controls:

- **Pulse Size:** 1-8 px per half-pulse.
- **Pulse Rate:** 10-60 complete pairs per second.
- **Ramp Mode:** `Instant` starts at the selected size; `Smooth` reaches it
  over the opening 150 ms.

The preset selector offers `Soft` (1 px, 20 Hz, Smooth), `Balanced` (2 px,
30 Hz, Smooth), and `Strong` (4 px, 45 Hz, Instant). `Custom` indicates a
combination that does not exactly match a preset. Horizontal output is always
zero.

## AI Aim mode

Choose `AI Aim` in the mode selector to use the fixed centered 320-by-320
capture and the bundled `models/all_games_320.onnx` model. AI Aim prefers the
nearest valid head detection; when no head is detected, it targets the upper
portion of the nearest valid player detection. It does not support arbitrary
models, training, overlays, or profiles.

AI Aim exposes four controls on the Motion page:

- **Confidence:** minimum accepted detection confidence, from 0.05 to 0.95.
- **Aim Strength:** movement scaling, from 0.05 to 2.00.
- **Smoothing:** interpolation strength, from 0.00 to 0.95.
- **Max Step:** maximum Makcu movement per update, from 1 to 127 counts.

Selecting AI Aim does not move the pointer. Enable arms capture and inference;
movement still requires the selected Trigger and optional Modifier. Status
shows `Ready (DirectML)` when DirectML is active or `Ready (CPU)` when the
explicit CPU fallback is in use. `STOP`, disable, disconnect, mode change, and
shutdown cancel AI inference and movement immediately. Trigger or Modifier
release stops movement while the armed AI capture remains ready.

## Requirements

- Windows 10 or newer
- Python 3.11+ with Tkinter installed
- A supported Makcu device and its USB driver for hardware operation
- A DirectML-capable Windows system for preferred AI inference; CPU fallback
  is available

Install the pinned runtime packages from a PowerShell prompt in this folder:

```powershell
python -m pip install -r requirements.txt
```

## Run from source

Normal feature work runs from source and does not build an executable:

```powershell
python main.py
```

You can also double-click `run_gui.bat`. Jitter starts disabled. Enable arms
the mover; actual movement occurs only while the selected Trigger and optional
Modifier are held. `Test 3s` uses the same motion engine without the trigger
gate and requires a connected Makcu. `STOP` immediately cancels movement and
test runs, including between paired half-pulses. Disconnect, hotkey disable,
Trigger release, and closing the window also signal an immediate stop. The
global hotkey (default `-`) toggles the enabled state once per press.

Without a connected Makcu, the UI remains usable for configuration but no
pointer movement can be sent. For a hardware check, connect the device before
launching, confirm the status changes to Connected, hold the configured
Trigger/Modifier, verify the up/down direction and the Soft, Balanced, and
Strong pulse sizes and rates, press `Test 3s`, exercise the hotkey, press
`STOP` between half-pulses, and verify that unplug/replug reconnects safely
and closing the window shuts the service down.

## User data and diagnostics

`config.json` and its backup `config.json.bak` are stored beside the source
script (or beside a packaged executable). Settings are loaded without
overwriting existing user data and writes are atomic. `app.log` in the same
folder contains timestamped diagnostics. These files are intentionally ignored
by Git.

## Verification

```powershell
python -m py_compile main.py ui.py motion.py ai_targeting.py ai_detection.py ai_capture.py ai_service.py makcu_service.py hotkeys.py settings.py liquid_widgets.py
python -m unittest discover -s tests -v
python -c "import makcu, onnxruntime, dxcam, numpy"
python -c "from ai_detection import OnnxDetector, model_resource_path; print(OnnxDetector(model_resource_path()).provider)"
Get-FileHash -Algorithm SHA256 models\all_games_320.onnx
git diff --check
```

The approved model SHA-256 is
`6B9157D6419F9DBC40D2DCECCC33A3387078C86F1C5872EDA544B174FF48499C`.

## License and source availability

Jitter and the bundled model are distributed under the GNU Affero General
Public License version 3. Every distributed executable or release must provide
access alongside it to the complete corresponding Jitter source for that exact
version, including build scripts and distribution metadata. A binary-only
release is not sufficient.

Bundled dependencies have separate obligations. Every release must include the
[third-party notices](THIRD_PARTY_NOTICES.md) and the complete
[release licensing checklist](licenses/README.md), including the exact license
files and GPL/LGPL source archives recorded in `licenses/manifest.json`, beside
the executable. Jitter's source alone does not satisfy every bundled
component's notice, corresponding-source, or relinking requirements.

## Explicit packaging

Packaging is opt-in. Normal development never builds an executable. When a
Windows one-file executable is specifically needed, run:

```powershell
gen.bat
```

The script installs Nuitka and its build helpers, runs the checks above, and
creates `build-output\Jitter.exe`; Nuitka output is recorded in
`build-output\build.log`. Use `gen.bat --help` to inspect this behavior without
installing dependencies or starting a build. Publishing that executable also
requires publishing or linking the matching complete source and all release
licensing materials described above. `gen.bat --review-json` exposes the
validated compile targets, imports, data options, and release materials without
installing dependencies or starting a build.
