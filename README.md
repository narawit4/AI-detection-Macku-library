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
nearest valid player detection.

The `MODEL` row starts with `Default · all_games_320.onnx`. `Browse...` can
select an external `.onnx` file for this process only, and `Use Default`
returns to the bundled model. A custom model must keep the exact `images`
`[1,3,320,320]` and `output0` `[1,300,6]` float contract and use class 0 for
players and class 7 for heads. Jitter validates it off the UI thread, pauses
AI during the switch, and after its exact ready event restarts the eligible AI
runtime and motion. A candidate startup failure makes one automatic rollback
attempt to restart the previous model. The selected path is never saved,
copied, packaged, or used by the release self-check; every launch starts with
the bundled model. Model changes are unavailable while `Test 3s` is active.
Jitter does not support training or profiles.

AI Aim exposes four controls on the Motion page:

- **Confidence:** minimum accepted detection confidence, from 0.05 to 0.95.
- **Aim Strength:** movement scaling, from 0.05 to 2.00.
- **Smoothing:** interpolation strength, from 0.00 to 0.95.
- **Max Step:** maximum Makcu movement per update, from 1 to 127 counts.

Below those controls, the five-point response curve maps target distance to
movement speed at `0%`, `25%`, `50%`, `75%`, and `100%` of the capture radius.
The zero point stays fixed; the other four nodes can be dragged or entered as
exact whole percentages, and must remain ordered from `0%` to `100%`. `Reset
Curve` restores the conservative `0 / 12 / 35 / 68 / 100` default. The curve
sets the distance response, Aim Strength scales it, Smoothing controls how
quickly velocity follows it, and Max Step caps each reported update. Curve
edits take effect continuously and are the only new persisted setting; tracker
history and runtime cadence are never saved.

Selecting AI Aim does not move the pointer. Master arms capture and inference;
movement still requires the selected Trigger and optional Modifier. Status
shows `Ready (DirectML)` when DirectML is active or `Ready (CPU)` when the
explicit CPU fallback is in use. `STOP` immediately cancels movement, hides the
Overlay, and ends its inference demand. Disable, disconnect, and source change
immediately cancel movement; inference remains active when the independent
visible Overlay still requires it. Shutdown ends both. Trigger or Modifier
release stops movement while the armed AI capture remains ready.

The independent `Overlay` control starts off. When enabled, it draws detection
boxes in a centered 320-by-320, click-through window that is excluded from
capture. `Box Color` changes the rectangle color, and `Head Boxes` can hide
head rectangles without changing AI Aim's head-first targeting. These two
display preferences persist, while Overlay visibility remains runtime-only.
Overlay viewing does not require AI Aim to be selected for movement; it starts
the approved detection runtime only while needed.

An AI runtime error fails closed by hiding the Overlay and deselecting AI Aim.
If Jitter remains selected under Master, Jitter continues or restarts through
the same gate; an AI-only failure disarms Master.

### Conservative tracking and movement

AI Aim tracks the complete detection box instead of choosing independently
from aim points on every frame. It predicts the current position from recent
motion, then considers same-class boxes inside a plausibility radius of
`max(48 px, 1.5 × the previous box diagonal)`. Candidates outside the
`0.4-2.5` area-ratio range are rejected; the remainder are ranked using
predicted distance, intersection-over-union, and area change.

The tracker can hold the last confirmed identity for up to 150 ms, but never
publishes that saved coordinate as movement. When plausible candidates are too
close to distinguish, the Overlay continues showing every current accepted box
without marking a provisional selection, while AI movement pauses. The
original target must be clear for two consecutive frames before movement
resumes. After the 150 ms hold expires, a replacement needs three stable
same-class observations before promotion. Provisional recovery and replacement
candidates are never published as AI movement targets.

Capture cadence follows the primary display refresh rate, capped at 240 FPS.
The movement servo runs at twice the detected display rate, clamped to the
120-480 Hz range. If refresh detection is unavailable or invalid, Jitter uses
the safe 120 FPS capture / 240 Hz servo fallback. The Motion page reports the
detected display and servo cadence, while measured inference FPS remains a
separate runtime status; none of these values are persisted.

One fresh AI target can be consumed as multiple time-based microsteps, so the
servo remains smooth between capture frames. Smoothing and acceleration are
computed from elapsed time, and an unconsumed target is discarded after 150
ms. These AI changes do not alter paired Jitter pulses, combined-source
composition, or the immediate STOP, disable, disconnect, source-change, and
shutdown cancellation rules.

### Adaptive Zoom

Adaptive Zoom is automatic and has no control or persisted setting. Every frame
still starts with the full-field 1.0× base pass. Only when a small target has
already been selected by that pass does AI Aim request a same-frame second pass
at 1.5× or 2.0×. The second pass is enabled only during connected, Master-armed,
AI-selected normal movement while the configured Trigger and Modifier (if any)
are active; it is not active while idle, during Overlay-only viewing, or during
`Test 3s`.

The `ZOOM` runtime status reports `1.0×`, `1.5×`, or `2.0×`. If refinement is
not eligible or does not produce a compatible result, the same frame keeps its
1.0× base result. A successful refinement replaces only the selected base box;
its coordinates are mapped back into the original 320-by-320 frame for the
Overlay, while unrelated base boxes remain. Adaptive Zoom does not magnify the
display and cannot recover a target that the base pass never detected. Zoom
status is runtime-only; Schema 5 and its persisted settings are unchanged.

Separately from full-box identity tracking, the recoil/zoom movement gate
requires two consecutive clear same-class base observations no more than 18
pixels apart. It may count a tracker's current clear recovery or replacement
candidate, but tracker publication remains authoritative: provisional targets
never move the pointer. Ambiguous or missing observations reset this separate
stability count. In combined mode, Jitter continues whenever that AI component
is withheld.

A new or shaken small target starts with the wider 1.5x refinement. A confirmed
target may return to 2.0x only after the fixed 100 ms recoil cooldown. A normal
refinement miss also restarts confirmation and cooldown while preserving only
the current frame's base boxes. These constants are internal runtime policy,
not saved settings.

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
Strong pulse sizes and rates, and that the optional configured-color centered
overlay is click-through and absent from capture.

## User data and diagnostics

`config.json` and its backup `config.json.bak` are stored beside the source
script (or beside a packaged executable). Schema 5 stores validated settings,
including the optional AI response curve; a missing or malformed curve loads
the complete safe default. Schemas 1-4 also use that default. Schema 6 and
newer files are treated as unsupported future data: Jitter runs with safe
in-memory defaults, disables saving, and leaves the file unchanged. Writes to
supported schemas are atomic. Overlay color and head-box visibility persist,
but motion-source selection, Master state, overlay visibility, tracker state,
targets, snapshots, FPS, provider, cadence, and zoom status are runtime-only.
`app.log` in the same folder contains timestamped diagnostics. These files are
intentionally ignored by Git.

## Verification

```powershell
python -m py_compile main.py ui.py motion.py combined_motion.py ai_targeting.py ai_tracking.py ai_detection.py ai_capture.py ai_zoom.py ai_service.py display_timing.py overlay.py makcu_service.py hotkeys.py settings.py sound_service.py liquid_widgets.py distribution_metadata.py
python -m unittest discover -s tests -v
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
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
