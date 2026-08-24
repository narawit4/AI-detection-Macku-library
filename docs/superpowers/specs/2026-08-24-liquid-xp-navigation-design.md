# Liquid XP Navigation Design

## Purpose

Redesign Jitter's fixed 640x560 dashboard around a top horizontal navigation
layer that combines a fluid active indicator with the existing Windows XP
Remastered visual language. The redesign must improve focus and organization
without changing motion behavior, Makcu behavior, configuration semantics, or
the always-visible emergency controls.

## Visual Direction

The navigation layer uses a rounded capsule, blue XP gradients, beveled
highlights, and a short sliding active pill. It evokes liquid navigation through
motion and shape rather than imitating platform-specific translucent blur that
Tkinter cannot render reliably. Light and dark palettes remain supported.

The content layer remains restrained and readable. Liquid styling is reserved
for navigation and its active state; content controls retain the established XP
Remastered cards, typography, spacing, focus treatment, and contrast.

## Page Structure

The window keeps its fixed 640x560 outer size and contains four vertical
regions:

1. A status strip for Makcu connection and device state.
2. A Liquid XP navigation bar.
3. A single-page content viewport.
4. A fixed footer followed by the fixed Enable/Runtime/STOP control row.

The navigation bar contains three tabs on the left and three persistent mini
icon actions on the right:

- `Setup`: Trigger, Modifier, Preset, Hotkey, and device setup information.
- `Motion`: Strength, Jitter Rate, Angle, and frequently used motion controls.
- `Advanced`: detailed numeric controls, Waveform, and Motion Curve.
- `↻`: Reconnect Makcu.
- `▶`: Test Run 3s.
- `☾` or `☀`: switch to Dark or Light Mode.

Reconnect, Test Run, and Theme leave the footer and Tools card and become one
persistent mini-icon group in the navigation bar. Each icon retains keyboard
focus and exposes an actionable tooltip. Advanced owns the only scrollable
content area. Setup and Motion fit without scrolling.

## Navigation Behavior

Clicking a tab selects its existing content frame. Left and Right arrow keys
move between adjacent tabs when navigation has focus. Only one page is visible
at a time, but all pages reuse the existing Tk variables and runtime bindings so
switching pages cannot reset settings or create duplicate callbacks.

The active pill animates to the selected tab over 160 ms.
Rapid selection cancels the obsolete animation and begins from the pill's
current rendered position. Navigation never delays STOP, disconnect handling,
hotkey disable, or shutdown. Animation callbacks are cancelled during teardown.

Changing pages does not enable, disable, start, or stop motion. Invalid numeric
input keeps the user on the current page and reports the existing concise footer
error. The UI never changes pages automatically in response to validation or
device events.

## Components and Responsibilities

### `LiquidXPNav` in `xp_widgets.py`

- Draw the capsule, tabs, active pill, focus state, and XP-style highlights.
- Own hit testing, keyboard traversal, animation state, and animation
  cancellation.
- Accept labels, selected index, palette, and a selection callback.
- Remain independent of Makcu, motion settings, configuration, and application
  lifecycle policy.
- Provide an explicit `cancel_animation()` method for shutdown.

### `JitterApp` in `ui.py`

- Build the fixed status, navigation, content, footer, and runtime regions.
- Own the Setup, Motion, and Advanced frames and show exactly one at a time.
- Map navigation selections to page frames.
- Place the three existing mini action buttons beside the custom navigation
  canvas while preserving their existing commands and state behavior.
- Apply palette changes to the navigation widget and cancel navigation activity
  before destroying Tk.

Motion, Makcu, hotkey, and settings modules require no interface or schema
changes for this redesign.

## Threading and Shutdown

All navigation drawing and animation run on Tk's main thread. Existing service
and hotkey workers continue to communicate through the thread-safe UI queue.
Shutdown first marks the application as closing, cancels scheduled UI work
including navigation animation and queue polling, signals runtime cancellation,
saves allowed configuration, and destroys the window. No worker calls Tk.

## Accessibility and Error Handling

Tabs expose visible text instead of icon-only navigation. The active state is
communicated through shape, contrast, and focus treatment rather than color
alone. Mini action buttons keep tooltips, keyboard focus, disabled state, and
their existing commands. Light and dark palettes must preserve readable text
and focus outlines.

Animation failures or unavailable geometry fall back to placing the active pill
directly at the selected tab. They must not prevent navigation or runtime
controls from working.

## Testing

Development follows TDD. Tests cover:

- the status/navigation/content/footer/runtime region order;
- Setup, Motion, and Advanced widget ownership;
- a single visible page and stable values across page switches;
- pointer and Left/Right keyboard selection;
- rapid-selection animation cancellation and final position;
- palette updates in Light and Dark Mode;
- the persistent `↻`, `▶`, and theme mini-icon group and tooltips;
- Test Run disabled-state behavior;
- STOP visibility on every page and while Advanced is scrolled;
- cancellation of animation and queue polling during shutdown;
- existing runtime, motion, settings, and hardware-free integration behavior.

Verification runs:

```powershell
python -m py_compile main.py ui.py motion.py makcu_service.py hotkeys.py settings.py
python -m unittest discover -s tests -v
python -c "import makcu"
```

No Nuitka build is part of ordinary implementation verification.

## Out of Scope

- Changing motion algorithms or presets.
- Changing Makcu protocol, reconnection, or button monitoring.
- Adding real transparency, blur dependencies, images, or non-standard UI
  packages.
- Changing the fixed window size.
- Adding tray, overlay, AI, profile, or training features.
