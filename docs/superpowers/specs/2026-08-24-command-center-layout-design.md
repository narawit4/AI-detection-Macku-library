# Command Center Layout Redesign

## Goal

Rebuild the complete Jitter page layout as a compact Windows XP Remastered
Command Center while preserving the fixed `640x560` window, existing runtime
behavior, and always-visible STOP control.

## Approved direction

The selected layout combines these approved choices:

- **A — Command Center:** a fixed setup column beside a primary work area;
- **A1 — Expand Under Quick:** Quick Jitter stays visible above expandable
  Advanced Settings;
- **G2 — Two-column Advanced grid:** Advanced numeric controls use two columns;
- Advanced Settings starts collapsed on every launch.

## Scope

This change restructures the Tk widget hierarchy and layout in `ui.py`. It does
not change motion generation, Makcu communication, hotkey behavior,
configuration keys or schema, validation limits, presets, save debounce, or
threading.

The existing Windows XP Remastered palette, Tahoma typography, styled buttons,
and `XPGlossySlider` component remain the visual foundation.

## Window structure

The native window remains titled `Jitter — Makcu Control`, fixed at `640x560`,
and non-resizable. Its content is divided vertically into four regions:

1. a thin fixed device/status strip;
2. a two-column Command Center body;
3. a fixed concise footer for status and actionable errors;
4. a fixed runtime action bar containing Enable, Runtime State, and STOP.

Neither the footer nor runtime action bar participates in scrolling. Expanding
or scrolling Advanced Settings must never move or obscure STOP.

## Device/status strip

The top strip presents the Makcu device label and connection state on one line.
Connection state retains the established colors: green for connected, amber for
connecting, and red for disconnected. Reconnect moves into the left setup
column so the status strip stays compact.

No internal brand banner is restored; branding remains in the native window
title.

## Command Center body

The body uses a fixed-width left column and a wider right column. Tk `grid`
weights define the split deterministically rather than using a user-resizable
paned window.

### Fixed left column

The left column remains visible while the right column scrolls. It contains:

- Trigger;
- Modifier;
- Preset;
- global Hotkey capture;
- Reconnect;
- Test 3s;
- the Advanced Settings expand/collapse control.

Related controls are grouped into compact XP-styled cards. Trigger, Modifier,
Preset, and Hotkey form the setup group; Reconnect, Test 3s, and the Advanced
toggle form the tools group.

### Scrollable right column

Only the right work area owns a Canvas and vertical scrollbar. Its embedded
content contains Quick Jitter first and Advanced Settings immediately after it.
Mouse-wheel routing must scroll this Canvas only while the pointer is over the
right work area and Advanced content exceeds the viewport.

Quick Jitter remains visible at the top when Advanced expands. It contains the
Strength and Jitter Rate controls with long horizontal XP Glossy sliders and
exact-value entries.

Advanced Settings starts collapsed. Expanding it inserts the existing Advanced
card under Quick Jitter without changing the outer geometry. Collapsing it
removes that card from layout and returns the right Canvas to the top.

## Advanced Settings grid

Advanced numeric controls retain the existing two-column arrangement and every
current key:

- Angle and Horizontal;
- Vertical;
- Randomness and Axis Phase;
- Smoothness and Ramp;
- Update Rate and Max Step;
- Acceleration and Deceleration.

Waveform and Motion Curve remain full-width choice rows below the numeric grid.
The Hotkey control moves to the fixed setup column and is not duplicated in
Advanced Settings.

Every numeric control retains its label, exact-value entry, public
`self.<key>_scale` and `self.<key>_entry` attributes, range, resolution,
validation trace, and `XPGlossySlider` interaction behavior.

## Runtime and footer

The runtime action bar uses three horizontal areas:

- a primary blue Enable/Disable Jitter button on the left;
- a centered runtime state and concise trigger hint in the middle;
- a high-contrast red STOP button on the right.

Enable and STOP have equal visual weight and remain fully visible at every
right-column scroll position. STOP retains its emergency-stop behavior and is
never nested under Advanced Settings.

The footer is a separate thin row immediately above the runtime action bar. It
continues to show concise actionable messages and errors. Detailed diagnostics
remain in `app.log` through the existing logging path.

## Interaction and data flow

This is a presentation-only redesign. Existing callbacks and data flow remain
unchanged:

- slider interaction updates the exact entry and immutable motion snapshot;
- valid exact-entry edits update the slider silently;
- applying a preset synchronizes variables and sliders;
- Enable arms movement, while Trigger and optional Modifier gate actual motion;
- Test 3s uses the production engine and remains interruptible;
- disconnect, hotkey disable, STOP, and shutdown retain immediate cancellation.

All Tk widget and variable access remains on the main thread. No new workers,
dependencies, image assets, or configuration fields are introduced.

## Visual treatment

- Preserve the existing warm XP window background and cool blue borders.
- Use white or warm-white card surfaces to separate functional groups.
- Make Quick Jitter the strongest visual focus in the work area.
- Keep spacing compact enough for `640x560` without reducing readability.
- Keep secondary controls light with dark text.
- Keep primary and danger controls white-on-blue and white-on-red.
- Preserve slider hover halo, pressed thumb, focus ring, and transient value
  bubble effects.

## Error and lifecycle behavior

Layout callbacks must tolerate window teardown and avoid leaving pending Tk
callbacks. Advanced expansion recalculates the right Canvas scroll region on
the Tk thread. Collapse resets the right Canvas to the top so Quick Jitter is
immediately visible.

Malformed numeric input continues to mark the corresponding exact entry and
leaves the last valid motion snapshot active. Layout changes do not alter save
or load error handling.

## Testing

Follow test-driven development. Update hardware-free UI tests to verify:

- the fixed `640x560` non-resizable geometry and native title;
- the status strip, left setup column, right work area, footer, and runtime bar
  exist in the approved order;
- Trigger, Modifier, Preset, Hotkey, Reconnect, Test 3s, and Advanced toggle are
  descendants of the fixed left column;
- Quick Jitter and Advanced Settings are descendants of the scrollable right
  content;
- Advanced starts collapsed, expands under Quick Jitter, and does not change
  the outer geometry;
- the Advanced numeric controls retain a two-column grid;
- scrolling the right work area does not move the left column, footer, runtime
  state, or STOP;
- every numeric control remains an `XPGlossySlider` with exact-entry and preset
  synchronization;
- current runtime, connection, shutdown, and configuration tests remain green.

After implementation run:

```powershell
python -m py_compile main.py ui.py xp_widgets.py motion.py makcu_service.py hotkeys.py settings.py
python -m unittest discover -s tests -v
python -c "import makcu"
```

Perform a Windows source UI smoke check without invoking movement. Verify the
column hierarchy, Advanced expand/collapse and right-only scrolling, slider
effects, preset synchronization, footer readability, fixed geometry, and STOP
visibility.

## Acceptance criteria

1. The application uses the approved Command Center hierarchy at `640x560`.
2. The left setup/tools column remains fixed and complete.
3. Quick Jitter remains at the top of the independently scrollable right work
   area.
4. Advanced Settings starts collapsed, expands below Quick Jitter, and keeps its
   approved two-column numeric grid.
5. Footer, Enable, Runtime State, and STOP remain fixed and fully visible.
6. Existing motion, Makcu, hotkey, configuration, validation, and slider
   behavior remains unchanged.
7. The complete hardware-free suite, module compilation, Makcu import, and
   Windows source UI smoke checks pass.
