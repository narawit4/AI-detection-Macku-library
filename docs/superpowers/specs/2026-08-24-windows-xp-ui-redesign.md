# Windows XP UI Redesign

## Goal

Redesign the standalone Jitter application's single-page interface around a
compact Windows XP Luna Blue visual language. The redesign changes presentation
and control placement only. Existing runtime, Makcu, motion, hotkey, persistence,
and shutdown behavior remains unchanged.

## Window and visual language

- Use a fixed `640x560` window and keep all text in English.
- Approximate Windows XP Luna Blue with application-owned Tkinter/ttk styles so
  the result remains consistent on supported modern Windows versions.
- Use Tahoma-compatible typography, a blue title treatment, warm XP-style panel
  surfaces, blue group-box labels, compact spacing, beveled controls, and clear
  focus states.
- Do not depend on the host operating system exposing an obsolete XP theme and
  do not introduce image or third-party UI dependencies.
- Connection state remains green, amber, or red for connected, connecting, and
  disconnected respectively.

## Information architecture

The always-visible compact dashboard contains these sections in order:

1. An application title area and Makcu connection summary with Reconnect.
2. A Runtime group showing the current runtime state, the primary Enable/Disable
   action, and the red emergency STOP action.
3. A Setup group containing Trigger, Modifier, Preset, and Test 3s.
4. A Quick Jitter group containing slider-plus-entry controls for Strength and
   Jitter Rate only.
5. An Advanced Settings toggle and a concise footer status/error message.

The STOP control stays in the always-visible Runtime group. Expanding Advanced
Settings must never move it outside the viewport.

## Advanced settings

Advanced Settings opens within the existing `640x560` window and uses the
existing internal scrolling mechanism. It contains:

- Hotkey capture
- Angle
- Horizontal jitter
- Vertical jitter
- Randomness
- Axis phase
- Smoothness
- Ramp time
- Update rate
- Maximum step
- Acceleration
- Deceleration
- Waveform
- Motion curve

Every numeric setting retains both its slider and exact-value input. Advanced
Settings is collapsed on launch.

## Interaction and state

- Jitter starts Disabled on every launch.
- Enable arms the existing mover; actual movement still requires the selected
  Trigger and Modifier when configured.
- Test 3s continues to use the production motion engine and stays interruptible
  by STOP or disconnect.
- Invalid exact values retain the last valid immutable motion snapshot, mark the
  relevant input, and show a concise footer error.
- Hotkey capture, preset application, configuration save debounce, reconnect,
  service event handling, and shutdown semantics do not change.
- All Tk access remains on the main thread.

## Implementation boundaries

Keep the redesign inside `ui.py` except for tests in `tests/test_ui.py`. Reuse
shared palette, spacing, font, and ttk style constants. Do not change the motion
engine, service interfaces, configuration schema, or add dependencies. Existing
public widget attributes used by tests and runtime integration should remain
available unless a test is intentionally updated for the approved placement.

## Testing and acceptance

Use test-driven development. First update or add UI tests that fail against the
old layout, then implement the redesign minimally. Tests must cover:

- fixed `640x560` geometry;
- Luna Blue style constants or widget styles;
- the compact always-visible control set;
- Hotkey and secondary motion controls inside Advanced Settings;
- STOP remaining visible while Advanced Settings is expanded;
- unchanged runtime actions and motion-setting bindings.

After implementation, run the repository verification commands:

```powershell
python -m py_compile main.py ui.py motion.py makcu_service.py hotkeys.py settings.py
python -m unittest discover -s tests -v
python -c "import makcu"
```

Perform a source UI smoke check on Windows to confirm the XP-inspired appearance,
internal scrolling, keyboard focus, readable control states, and uninterrupted
STOP access. Hardware behavior requires a connected Makcu device and is outside
hardware-free automated verification.
