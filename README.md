# Jitter

Jitter is a small Windows-only Tkinter controller for a Makcu USB device. It
has a fixed-size English Liquid Control Deck with Control, Motion, and Advanced
pages: configure two-dimensional smooth jitter, choose a Trigger and optional
Modifier, arm or stop movement, run a three-second test, and assign a global
toggle hotkey.

## Requirements

- Windows 10 or newer
- Python 3.11+ with Tkinter installed
- A supported Makcu device and its USB driver for hardware operation

Install the only runtime package from a PowerShell prompt in this folder:

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
gate. `STOP` immediately cancels movement and test runs. The global hotkey
(default `-`) toggles the enabled state once per press.

Without a connected Makcu, the UI remains usable for configuration but no
pointer movement can be sent. For a hardware check, connect the device before
launching, confirm the status changes to Connected, hold the configured
Trigger/Modifier, press Test 3s, exercise the hotkey, press STOP, and verify
that unplug/replug reconnects safely.

## User data and diagnostics

`config.json` and its backup `config.json.bak` are stored beside the source
script (or beside a packaged executable). Settings are loaded without
overwriting existing user data and writes are atomic. `app.log` in the same
folder contains timestamped diagnostics. These files are intentionally ignored
by Git.

## Verification

```powershell
python -m py_compile main.py ui.py motion.py makcu_service.py hotkeys.py settings.py
python -m unittest discover -s tests -v
python -c "import makcu"
```

## Explicit packaging

Packaging is opt-in. Normal development never builds an executable. When a
Windows one-file executable is specifically needed, run:

```powershell
gen.bat
```

The script installs Nuitka and its build helpers, runs the checks above, and
creates `build-output\Jitter.exe`; Nuitka output is recorded in
`build-output\build.log`. Use `gen.bat --help` to inspect this behavior without
installing dependencies or starting a build.
