# Jitter

Jitter is a small Windows-only Tkinter controller for a Makcu USB device. It
has a fixed-size English Liquid Split Console with Control, Motion, and
Settings pages.
It sends paired pulses on an axis tilted 45 degrees to the right of vertical:
one equal pair moves up-right then down-left, and the following equal pair
moves down-left then up-right; this two-pair order repeats. Complete pulse
pairs have zero intended displacement; results vary with the receiving
application's input processing. Choose a Trigger and optional Modifier, arm or
stop movement, run a three-second test, and assign a global toggle hotkey.

The `Jitter` and `AI Aim` source buttons are independent: select either source
or both. Master (and the global hotkey) arms the currently selected sources;
movement still requires the configured Trigger and optional Modifier. When both
are selected, their current two-dimensional deltas are summed. Jitter keeps
running when AI Aim has no target.

## Jitter motion controls

In Jitter mode, the Motion page provides these controls:

- **Pulse Size:** 1-8 px per half-pulse.
- **Pulse Rate:** 20-120 complete pairs per second.
- **Ramp Mode:** `Instant` starts at the selected size; `Smooth` reaches it
  over the opening 150 ms.

The preset selector offers `Soft` (1 px, 30 Hz, Smooth), `Balanced` (2 px,
60 Hz, Smooth), and `Strong` (4 px, 100 Hz, Instant). `Custom` indicates a
combination that does not exactly match a preset. Horizontal and vertical
components are equal in magnitude along the 45-degree pulse axis.

## AI Aim source and overlay

Select `AI Aim` to use the fixed centered 320-by-320 capture and the bundled
`models/all_games_320.onnx` model. AI Aim prefers the nearest valid head
detection; when no head is detected, it targets the upper portion of the
nearest valid player detection. It does not support arbitrary models, training,
or profiles.

AI Aim exposes four controls on the Motion page:

- **Confidence:** minimum accepted detection confidence, from 0.05 to 0.95.
- **Aim Strength:** movement scaling, from 0.05 to 2.00.
- **Smoothing:** interpolation strength, from 0.00 to 0.95.
- **Max Step:** maximum Makcu movement per update, from 1 to 127 counts.

Selecting AI Aim does not move the pointer. Master arms capture and inference;
movement still requires the selected Trigger and optional Modifier. Status
shows `Ready (DirectML)` when DirectML is active or `Ready (CPU)` when the
explicit CPU fallback is in use. `STOP` immediately cancels movement, hides the
Overlay, and ends its inference demand. Disable, disconnect, and source change
immediately cancel movement; inference remains active when the independent
visible Overlay still requires it. Shutdown ends both. Trigger or Modifier
release stops movement while the armed AI capture remains ready.

The independent `Overlay` control starts off. When enabled, it draws red
detection boxes in a centered 320-by-320, click-through window that is excluded
from capture. Overlay viewing does not require AI Aim to be selected for
movement; it starts the approved detection runtime only while needed.

An AI runtime error fails closed by hiding the Overlay and deselecting AI Aim.
If Jitter remains selected under Master, Jitter continues or restarts through
the same gate; an AI-only failure disarms Master.

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

You can also double-click `run_gui.bat`. Jitter and AI Aim both start
unselected, and Master starts disabled. Master and the global hotkey (default
`-`) arm or disable the selected sources; actual movement occurs only while the
selected Trigger and optional Modifier are held. `Test 3s` follows the sources
selected when the test starts, temporarily bypasses the Trigger gate, and
requires a connected Makcu. `STOP` immediately cancels movement and test runs,
including between paired half-pulses. Disconnect, hotkey disable, Trigger
release, and closing the window also signal an immediate stop.
The Settings page can mute the hotkey ON/OFF cues, set their volume from
0–100, and preview either cue without changing the armed state.

Without a connected Makcu, the UI remains usable for configuration but no
pointer movement can be sent. Hardware-dependent behavior still requires a
connected-device check: verify each selected-source combination with the
configured Trigger/Modifier, Test 3s with those sources, the global hotkey,
STOP between half-pulses, disconnect/reconnect, and shutdown. Also verify the
diagonal up-right/down-left paired Jitter direction, the Soft, Balanced, and
Strong pulse sizes and rates, and that the optional red centered overlay is
click-through and absent from capture.

## User data and diagnostics

`config.json` and its backup `config.json.bak` are stored beside the source
script (or beside a packaged executable). Schema 4 stores validated settings;
settings are loaded without overwriting existing user data and writes are
atomic. Motion-source selection, Master state, overlay visibility, targets,
snapshots, FPS, and provider status are runtime-only. `app.log` in the same
folder contains timestamped diagnostics. These files are intentionally ignored
by Git.

## Verification

```powershell
python -m py_compile main.py ui.py motion.py combined_motion.py ai_targeting.py ai_detection.py ai_capture.py ai_service.py overlay.py makcu_service.py hotkeys.py settings.py sound_service.py liquid_widgets.py distribution_metadata.py
python -m unittest discover -s tests -v
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
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
Windows one-file executable is specifically needed, run the no-argument
compatibility entry and type the exact confirmation word `BUILD`:

```powershell
.\gen.bat
```

For exact argument-bearing help, review, or deliberate non-interactive build
automation, invoke the standard-library Python launcher directly:

```powershell
python .\distribution_metadata.py --help
python .\distribution_metadata.py --review-json
python .\distribution_metadata.py --build
```

The build installs Nuitka and its build helpers, runs the checks above, and
creates `build-output\Jitter.exe`; Nuitka output is recorded in
`build-output\build.log`. Publishing that executable also requires publishing
or linking the matching complete source and all release licensing materials
described above.

The canonical Nuitka command loads `nuitka-package.config.yml` so the pinned
ONNX Runtime DirectML `onnxruntime/capi/DirectML.dll` is bundled explicitly.
After Nuitka finishes, the build runs this exact non-GUI check before copying
release materials:

```powershell
.\build-output\Jitter.exe --ai-runtime-self-check
```

It prints JSON containing the model path, SHA-256, and active provider, and it
fails unless the approved model contract and hash pass with
`DmlExecutionProvider`. The source-mode equivalent is
`python .\main.py --ai-runtime-self-check`.

`gen.bat` intentionally has no argument interface and never reads or forwards
batch arguments. Do not append arguments to it: `cmd.exe` parses quotes,
metacharacters, pipes, and redirects before a batch file can validate them, so
shell text such as `>file` is owned by the calling shell. Pass constructed or
untrusted argument vectors directly to Python without a shell; invalid, empty,
or extra launcher arguments are then rejected before any build plan executes.
