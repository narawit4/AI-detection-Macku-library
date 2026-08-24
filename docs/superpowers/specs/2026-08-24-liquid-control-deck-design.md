# Liquid Control Deck UI Redesign

## Goal

Replace the Windows XP-inspired Jitter interface with a complete modern Liquid
Control Deck while preserving all device, motion, configuration, hotkey, Test
Run, emergency-stop, and shutdown behavior.

The interface remains an English-only, fixed-size Windows Tkinter application.
It supports complete dark and light themes and does not depend on image assets
or additional runtime packages.

## Visual Direction

The application uses a calm layered liquid-glass language rather than Windows
XP styling. A softly graded application background supports floating panels
with rounded geometry, translucent-looking tonal layers, thin borders, and
restrained highlights. Tkinter does not provide real blur or alpha-composited
child widgets, so the effect is rendered deterministically with opaque blended
colors and Canvas primitives.

Dark and light modes use the same spacing, geometry, hierarchy, and interaction
states. Each theme supplies dedicated surface, raised-surface, border, text,
muted-text, accent, hover, pressed, focus, and danger colors. Connection status
continues to use green, amber, and red. Primary cyan controls always use dark
readable text; secondary controls remain legible in every state.

The window is approximately `780x640`, fixed-size, and does not change size
when Advanced content is shown.

## Page Structure

The shell contains five persistent vertical regions:

1. A compact identity row with `Jitter`, a concise subtitle, and a luminous
   Makcu connection indicator.
2. A full-width Liquid Navigation command bar.
3. A page host containing exactly one visible page.
4. A persistent runtime dock containing Enable, runtime state, and STOP.
5. A concise footer for actionable errors and status messages.

The Liquid Navigation command bar contains three destinations on the left:
`Control`, `Motion`, and `Advanced`. A rounded selection lens glides between
destinations using a short cancellable animation. On the right, mini icon
buttons provide `Reconnect`, `Test 3s`, and theme switching. Their tooltips and
accessible text identify their actions; they remain available on every page.

The page allocation is:

- **Control:** Trigger, optional Modifier, preset selection, global hotkey, and
  a concise device summary.
- **Motion:** Strength and Jitter Rate quick controls plus the essential live
  motion summary.
- **Advanced:** waveform, motion curve, and all remaining numeric controls in
  an internally scrollable region.

Changing pages does not rebuild page widgets or alter settings. The current
page is presentation state only and is not persisted.

## Persistent Runtime Dock

The runtime dock is visible on every page and anchors the safety hierarchy.
Enable occupies the left side, the current `DISABLED`, `ARMED`, `MOVING`, or
`TESTING` state is centered, and the red STOP control occupies the right side.
STOP remains visually dominant and immediately signals cancellation; it never
waits for a normal movement interval.

Jitter remains Disabled on every launch. Enable only arms movement. Normal
movement still requires the configured Trigger and optional Modifier. Test 3s
continues to use the production motion engine, temporarily bypasses Trigger,
requires Makcu, and remains interruptible by STOP or disconnect.

## Liquid Components

Rename the reusable widget module from `xp_widgets.py` to
`liquid_widgets.py`. Remove XP terminology from application-owned classes,
style names, palette constants, tests, and active documentation references.
Historical design and plan documents remain unchanged as project history.

The module provides:

- `LiquidNavigation`: keyboard-accessible navigation with mouse selection,
  arrow-key movement, cancellable lens animation, theme palette updates, and
  safe destruction.
- `LiquidSlider`: the existing exact slider behavior rendered with a new
  rounded rail, accent fill, soft thumb halo, hover/pressed/focus feedback, and
  transient value bubble.
- `LiquidIconButton` or an equivalent isolated Canvas component: a compact,
  rounded action button with text-symbol icon, hover, pressed, focus, disabled,
  keyboard activation, tooltip metadata, and theme palette updates.

Icons use Unicode or Canvas line geometry that is reliable with the supported
Windows fonts. No raster assets, Pillow, or new dependencies are introduced.

## Theme and Interaction Behavior

Theme switching updates every live surface and custom Canvas widget without
recreating the application or losing focus, page selection, or setting values.
The existing persisted `theme` setting remains `dark` or `light`.

All custom controls expose visible keyboard focus. Navigation supports Left,
Right, Home, End, Space, and Enter where appropriate. Icon actions support
Space and Enter. Hover and pressed effects supplement rather than replace
focus and disabled indicators.

Animations are short and decorative. Every scheduled callback is tracked,
cancellable, generation-safe where needed, and stopped before widget teardown.
If Tk scheduling is unavailable during shutdown, the widget snaps to its final
state without raising an error.

## Architecture and Data Flow

`ui.py` remains the owner of Tk variables, page construction, application
state, service-event polling, settings updates, and shutdown. It maps the
active theme to shared palette dictionaries and passes those dictionaries to
custom liquid widgets.

`liquid_widgets.py` remains presentation-only. Its widgets accept values and
callbacks but do not import motion, settings, Makcu, or hotkey services.

Device callbacks and workers continue to marshal UI events through the current
queue and `after` mechanisms. Motion snapshots, connection generations,
emergency stop, reconnect, and shutdown retain their current behavior. No Tk
widget or Tk variable is accessed from a worker thread.

## Error Handling and Shutdown

Concise actionable errors remain in the footer. Detailed diagnostics continue
to use thread-safe logging in `app.log`.

Reconnect, Test 3s, theme switching, navigation animation, hotkey capture, and
advanced scrolling must tolerate shutdown already being in progress. Closing
the window cancels all UI animation callbacks before destroying widgets and
then performs the existing bounded service cleanup. The redesign must not add
blocking work to the Tk event loop.

## Testing

Development follows test-driven development. Tests first define the new
contract and fail against the XP interface, then production code is changed
minimally.

Coverage includes:

- Liquid navigation selection, keyboard behavior, palette changes, animation
  replacement/cancellation, and shutdown-safe destruction.
- Liquid slider value semantics, pointer and keyboard interaction, palette
  changes, focus, disabled behavior, and callback compatibility.
- Mini icon actions, accessible metadata, keyboard activation, and tooltips.
- Shell region order, page ownership, Control/Motion/Advanced allocation,
  persistent mini actions, runtime dock, and STOP visibility.
- Complete dark/light theme propagation with readable button and status colors.
- Advanced internal scrolling without outer-window resizing.
- Existing settings synchronization, reconnect, Test Run, queue draining,
  STOP, and shutdown tests.

Final verification runs the repository-required compile command, the complete
unit-test suite, `python -c "import makcu"`, and `git diff --check`. A connected
Makcu device is still required for hardware verification and is reported
separately rather than simulated.

## Acceptance Criteria

1. No active UI surface, class, style, palette, or source test presents Windows
   XP styling or naming.
2. The fixed-size dashboard visibly follows the Liquid Control Deck hierarchy
   and supports both complete themes.
3. Control, Motion, and Advanced navigation is keyboard-accessible and does not
   lose settings or recreate controls.
4. Reconnect, Test 3s, and theme switching remain visible as mini icon actions
   on every page.
5. Enable, runtime status, and STOP remain visible on every page, with STOP
   immediately interrupting movement or Test 3s.
6. Advanced content scrolls inside the existing window.
7. Closing the window remains bounded and does not hang because of new UI
   callbacks or animations.
8. All hardware-free tests and required static verification commands pass.
