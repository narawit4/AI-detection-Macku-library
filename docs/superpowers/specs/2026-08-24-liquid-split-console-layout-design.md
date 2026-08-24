# Liquid Split Console Layout Redesign

## Goal

Replace the Liquid Control Deck's horizontal page hierarchy with a compact
`840x620` Liquid Split Console. Preserve the established liquid visual system,
complete dark/light themes, and every runtime, safety, settings, motion,
Makcu, hotkey, Test Run, and shutdown behavior.

## Shell Structure

The fixed-size window contains two primary columns:

1. A persistent `176` pixel floating navigation rail on the left.
2. A flexible console workspace on the right.

The console workspace contains the active page, a concise footer, and the
persistent runtime dock. The footer and dock never scroll. The outer window
does not resize when pages change or Advanced scrolls.

The shell keeps deterministic Canvas-rendered gradient bands, rounded floating
surfaces, thin highlights, and opaque blended colors. Tkinter does not provide
real child-widget blur, so no true transparency or additional image dependency
is introduced.

## Navigation Rail

The rail is a vertical Liquid Navigation control rather than rotated top tabs.
It contains:

- Jitter identity and a concise `MAKCU MOTION` subtitle at the top.
- A glowing Makcu connection orb with adjacent connection text.
- Three vertically stacked destinations: `Control`, `Motion`, and `Advanced`.
- Three mini icon actions anchored at the bottom: Reconnect, Test 3s, and
  theme switching.

A rounded selection lens glides vertically between destinations. Pointer,
Up/Down, Home/End, Enter, and Space interactions remain supported. Reconnect,
Test, and theme actions remain visible on every page and retain tooltips,
accessible names, focus, disabled, and keyboard states.

## Console Pages

Only one persistent page is visible at a time. Page changes do not recreate
widgets, modify settings, or persist presentation state.

### Control

Control uses an asymmetric two-column layout:

- The wider left card contains Trigger, Modifier, and global Hotkey.
- The narrower right card contains the Makcu device summary and Preset.

The connection orb in the rail is the primary connection indicator; the device
card provides details without duplicating a large status header.

### Motion

Motion uses a hero-control layout:

- The wider left column contains large Strength and Jitter Rate sliders.
- The narrower right card contains the current immutable Live Snapshot.

The snapshot continues to summarize strength, angle, horizontal/vertical
jitter, rate, waveform, and smoothness. It refreshes after valid edits and
presets and remains unchanged after invalid edits.

### Advanced

Advanced uses the existing two-column numeric grid and waveform/curve choices
inside a page-local scrollable region. Its scrollbar and combobox popdowns
remain completely themed. Scrolling affects only Advanced.

## Footer and Runtime Dock

The footer is a thin actionable-message strip directly above the runtime dock.
Detailed diagnostics remain in `app.log`.

The floating runtime dock spans the workspace bottom:

- Enable on the left.
- Exact uppercase `DISABLED`, `ARMED`, `MOVING`, or `TESTING` in the center.
- The visually dominant red STOP action on the right.

The dock remains visible on every page and while Advanced is scrolled. STOP
continues to signal cancellation immediately without waiting for a movement
interval.

## Themes and Accessibility

Dark and light themes share identical geometry and hierarchy. Existing palette
roles remain the source of truth for shell, cards, navigation, sliders, icon
actions, entries, comboboxes, scrollbar, footer, runtime dock, and connection
orb. Existing contrast thresholds remain enforced: normal text at least 4.5:1
and icon states at least 3:1.

Navigation and all actions expose visible keyboard focus. Tab traversal reaches
the rail, mini actions, page controls, footer-independent runtime actions, and
STOP in a predictable order. Disabling and re-enabling Test 3s preserves real
keyboard focus when Tk retains it.

## Architecture

`ui.py` continues to own Tk variables, application state, service-event
marshalling, page construction, palette mapping, settings updates, and bounded
shutdown. It rearranges widget parents and geometry only; existing runtime
methods and service boundaries remain unchanged.

`liquid_widgets.py` extends `LiquidNavigation` with an orientation parameter:

```python
LiquidNavigation(
    parent,
    *,
    labels: tuple[str, ...],
    command,
    palette=None,
    orientation: str = "horizontal",
)
```

`orientation` accepts only `horizontal` or `vertical`. Horizontal behavior
remains compatible; vertical mode maps Up/Down to adjacent destinations,
renders a vertically moving lens, and retains Home/End/Enter/Space behavior.
The widget module remains presentation-only and does not import application
services.

## Shutdown and Error Handling

All navigation animation, slider bubble, tooltip, icon, queue, save, and
hotkey-capture callbacks remain owned and cancelled before Tk teardown.
Expected scheduler `tk.TclError` is handled only around scheduling operations;
drawing and contract errors are not hidden.

Closing the window continues to set the closing state first, immediately stop
motion, stop the hotkey watcher, close Makcu with bounded cleanup, save safe
configuration, and destroy Tk without waiting on normal movement timing.

## Testing

Development follows TDD. Tests first fail against the horizontal Control Deck,
then production code changes minimally.

Coverage includes:

- Vertical navigation geometry, animation, keyboard mapping, activation,
  palette propagation, scheduler failure, and destruction.
- Exact `840x620` geometry and two-column shell ownership.
- Persistent rail contents and bottom mini-action placement.
- Control asymmetric card ownership.
- Motion hero controls and Live Snapshot placement/synchronization.
- Advanced-only internal scrolling.
- Footer/runtime order and persistent STOP visibility.
- Both themes, connection states, contrast, focus, and disabled behavior.
- Existing runtime, settings, service queue, Test Run, emergency stop, and
  shutdown regression tests.

Final verification runs the repository-required compile command, the complete
unit suite, `python -c "import makcu"`, and `git diff --check`. A source smoke
check covers both themes and pages without invoking movement. Physical Makcu
behavior remains a separate hardware verification boundary. Nuitka is not run
unless explicitly requested.

## Acceptance Criteria

1. The fixed window is `840x620` with a persistent left navigation rail and
   right console workspace.
2. Control, Motion, and Advanced use the approved layouts without rebuilding
   controls or changing settings.
3. Reconnect, Test 3s, theme switching, connection state, footer, runtime state,
   Enable, and STOP remain visible in their persistent regions.
4. Advanced alone scrolls; the footer and runtime dock remain fixed.
5. Dark/light themes and accessibility thresholds remain complete.
6. Motion, Makcu, hotkey, settings, Test Run, STOP, and shutdown behavior do not
   regress.
7. All hardware-free verification passes and the window closes promptly.
