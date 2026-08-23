# XP Glossy Slider Design

## Goal

Replace every numeric `tk.Scale` in Quick Jitter and Advanced Settings with a
reusable Windows XP-inspired glossy slider that provides clear hover, focus,
drag, and keyboard feedback without changing motion behavior or configuration.

## Scope

The slider applies to every numeric setting currently created through
`JitterApp._numeric_control()`. Exact-value entries remain beside their sliders.
Preset application, validation, immutable motion snapshots, save debounce,
Makcu movement, runtime controls, and configuration schema remain unchanged.

## Component boundary

Create `xp_widgets.py` containing `XPGlossySlider`, a reusable `tk.Canvas`
subclass. The component owns rendering, value/position conversion, pointer and
keyboard interaction, and its transient value bubble. It does not import UI,
motion, Makcu, hotkey, or configuration modules.

Constructor interface:

```python
XPGlossySlider(
    parent,
    *,
    from_: float,
    to: float,
    resolution: float,
    command: Callable[[str], None] | None = None,
    width: int = 220,
    height: int = 34,
)
```

Public interface:

```python
slider.get() -> float
slider.set(value: float) -> None
```

`set()` clamps and snaps to the configured range and resolution, redraws the
component, and does not invoke `command`. Pointer and keyboard interaction calls
`command` with the formatted numeric value after updating the component.

## Visual design

- The unfilled rail uses a warm light-gray XP surface with a cool gray border.
- The filled rail uses a left-to-right light-to-deep blue glossy treatment.
- The thumb uses layered highlight, body, border, and shadow shapes to suggest a
  raised XP control without bitmap assets.
- Hover adds a restrained blue halo around the thumb.
- Pressing darkens and visually depresses the thumb.
- Keyboard focus adds the existing gold XP focus color without relying on color
  alone; the outline thickness also changes.
- A compact value bubble appears above the thumb on pointer hover or drag and
  disappears shortly after pointer leave or release.
- Rendering uses only Tk Canvas primitives and application-owned colors. No
  Pillow, image files, third-party UI library, or animation worker is added.

Canvas has no native gradient primitive, so each glossy gradient is rendered as
a small fixed number of adjacent colored rectangles. Rendering is bounded and
occurs only on resize, state transition, or value change.

## Interaction

- Clicking or dragging anywhere on the rail sets the value nearest the pointer.
- Values are clamped to the inclusive `[from_, to]` range and snapped to the
  configured resolution.
- Left/Down decrement one resolution step; Right/Up increment one step.
- Home selects the minimum and End selects the maximum.
- Tab focus works through normal Tk focus traversal.
- Pointer capture keeps a drag coherent until button release.
- The bubble show/hide delay uses Tk `after()` callbacks only. Pending callbacks
  are cancelled when superseded and tolerate widget destruction.
- Programmatic `set()` remains silent so UI synchronization cannot recursively
  invoke `_scale_changed()`.

## Value formatting

Values whose snapped result is mathematically integral are displayed and sent
to `command` without a decimal suffix. Other values use the minimum decimal
precision required by `resolution`, without binary floating-point noise.

## Jitter UI integration

`ui.py` imports `XPGlossySlider` and replaces the `tk.Scale` construction inside
`_numeric_control()`. Existing attributes such as `motion_strength_pps_scale`
continue to reference the slider instance. Existing calls to `set()`, motion
variable traces, exact-entry validation, and `_scale_changed(key, value)` remain
compatible.

The current fixed `640x560` window, native title, fixed Runtime/STOP area, Setup,
internal Advanced scrolling, and XP Remastered buttons do not change.

## Error and lifecycle handling

Invalid constructor ranges or non-positive resolution raise `ValueError`.
Pointer coordinates outside the rail clamp safely. Destroying the component
cancels its pending bubble callback. Rendering and callbacks remain on the Tk
main thread.

## Testing

Add `tests/test_xp_widgets.py` for hardware-free component tests covering:

- minimum, midpoint, and maximum value-position conversion;
- clamping and resolution snapping;
- silent programmatic `set()`;
- pointer drag command emission;
- arrow and Home/End keyboard behavior;
- hover, pressed, focused, and bubble state transitions;
- destruction with a pending bubble callback.

Update `tests/test_ui.py` to confirm all numeric controls are
`XPGlossySlider` instances, exact entries still synchronize both directions,
presets still update sliders, STOP remains visible after Advanced scrolling,
and outer geometry remains `640x560`.

Follow TDD for the pure conversion behavior, component interaction, and UI
integration. After implementation run:

```powershell
python -m py_compile main.py ui.py xp_widgets.py motion.py makcu_service.py hotkeys.py settings.py
python -m unittest discover -s tests -v
python -c "import makcu"
```

Perform a Windows source UI smoke check for hover, dragging, keyboard focus,
value bubble readability, Advanced scrolling, preset synchronization, and STOP
visibility. Nuitka packaging is outside this change.

## Acceptance criteria

1. Every Quick Jitter and Advanced numeric slider uses `XPGlossySlider`.
2. All sliders show the approved glossy XP rail, thumb, hover, pressed, focus,
   and transient value-bubble feedback.
3. Mouse, keyboard, exact entry, and preset changes remain synchronized.
4. Motion settings, validation, persistence, runtime behavior, and threading
   semantics remain unchanged.
5. The fixed window and always-visible STOP requirement continue to hold.
6. The complete hardware-free suite and Makcu import pass.
