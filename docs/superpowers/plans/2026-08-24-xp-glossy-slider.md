# XP Glossy Slider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every numeric Tk scale with a reusable XP Glossy slider that supplies mouse, keyboard, hover, pressed, focus, and transient value-bubble feedback while preserving Jitter behavior.

**Architecture:** Add an isolated `XPGlossySlider` Canvas component in `xp_widgets.py`, first implementing deterministic value semantics and then Canvas rendering/input behavior. Integrate it only through `JitterApp._numeric_control()` so every current Quick Jitter and Advanced numeric control adopts it without changing motion, configuration, or runtime services.

**Tech Stack:** Python 3, Tkinter Canvas, `unittest`

**Spec:** `docs/superpowers/specs/2026-08-24-xp-glossy-slider-design.md`

## Global Constraints

- Windows-only fixed-size English Tkinter desktop application.
- Keep the root geometry exactly `640x560` and non-resizable.
- Keep Runtime and STOP fixed outside the scrolling Canvas and always visible.
- Apply `XPGlossySlider` to every numeric control created through `_numeric_control()` in Quick Jitter and Advanced Settings.
- Retain an exact-value entry beside every slider.
- Preserve each existing numeric key, minimum, maximum, resolution, public `*_scale` attribute, callback, variable trace, validation path, preset behavior, and save debounce.
- `XPGlossySlider.set(value)` clamps/snaps/redraws without invoking `command`; pointer and keyboard changes invoke `command` after the value changes.
- Keep all Tk and Canvas access on the main thread; use only Tk `after()` for delayed bubble hiding.
- Do not import motion, UI, Makcu, hotkey, or settings modules from `xp_widgets.py`.
- Do not add Pillow, image assets, third-party UI libraries, workers, or runtime dependencies.
- Preserve the current uncommitted XP Remastered button, removed internal banner, native `Jitter — Makcu Control` title, and their tests in `ui.py` and `tests/test_ui.py`.
- Do not run Nuitka or edit generated output.

## File Structure

- Create `xp_widgets.py`: self-contained Canvas slider, rendering, interaction, value formatting, and callback lifecycle.
- Create `tests/test_xp_widgets.py`: hardware-free component semantics, input, Canvas-state, and teardown tests.
- Modify `ui.py`: import and construct the component inside `_numeric_control()` only.
- Modify `tests/test_ui.py`: verify complete slider adoption and existing synchronization behavior.

---

### Task 1: Implement deterministic slider value semantics

**Files:**
- Create: `xp_widgets.py`
- Create: `tests/test_xp_widgets.py`

**Interfaces:**
- Consumes: `tk.Canvas`, numeric `from_`, `to`, `resolution`, and optional `Callable[[str], None]`.
- Produces: `XPGlossySlider(parent, *, from_, to, resolution, command=None, width=220, height=34)`, `get() -> float`, `set(value: float) -> None`, `_snap(value: float) -> float`, `_format_value(value: float) -> str`, `_value_to_x(value: float) -> float`, and `_x_to_value(x: float) -> float`.

- [ ] **Step 1: Write failing construction and value tests**

Create `tests/test_xp_widgets.py` with a real withdrawn Tk root:

```python
import tkinter as tk
import unittest

from xp_widgets import XPGlossySlider


class _SliderTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self):
        self.root.destroy()

    def make_slider(self, **kwargs):
        options = {"from_": 0, "to": 100, "resolution": 1,
                   "width": 220, "height": 34}
        options.update(kwargs)
        slider = XPGlossySlider(self.root, **options)
        slider.pack()
        self.root.update_idletasks()
        return slider


class XPGlossySliderValueTests(_SliderTestCase):

    def test_rejects_invalid_range_and_resolution(self):
        with self.assertRaises(ValueError):
            self.make_slider(from_=1, to=1)
        with self.assertRaises(ValueError):
            self.make_slider(resolution=0)

    def test_set_clamps_and_snaps_without_calling_command(self):
        emitted = []
        slider = self.make_slider(from_=0.1, to=60, resolution=0.1,
                                  command=emitted.append)
        slider.set(22.26)
        self.assertAlmostEqual(slider.get(), 22.3)
        slider.set(-5)
        self.assertAlmostEqual(slider.get(), 0.1)
        slider.set(100)
        self.assertAlmostEqual(slider.get(), 60.0)
        self.assertEqual(emitted, [])

    def test_formats_integral_and_fractional_resolutions_cleanly(self):
        integer = self.make_slider(resolution=1)
        decimal = self.make_slider(from_=0.1, to=60, resolution=0.1)
        self.assertEqual(integer._format_value(22), "22")
        self.assertEqual(decimal._format_value(22.3), "22.3")

    def test_value_position_conversion_covers_range_endpoints_and_midpoint(self):
        slider = self.make_slider()
        left, right = slider._rail_bounds()
        self.assertAlmostEqual(slider._value_to_x(0), left)
        self.assertAlmostEqual(slider._value_to_x(50), (left + right) / 2)
        self.assertAlmostEqual(slider._value_to_x(100), right)
        self.assertEqual(slider._x_to_value(left - 50), 0)
        self.assertEqual(slider._x_to_value(right + 50), 100)
```

- [ ] **Step 2: Run the new test module and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_xp_widgets.py -v
```

Expected: import error because `xp_widgets.py` does not exist.

- [ ] **Step 3: Implement the minimal Canvas class and numeric helpers**

Create `xp_widgets.py` with constructor validation, silent `set()`, clean Decimal-based snapping, and coordinate conversion:

```python
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import math
import tkinter as tk
from typing import Callable


class XPGlossySlider(tk.Canvas):
    THUMB_RADIUS = 8
    RAIL_INSET = 14

    def __init__(self, parent, *, from_: float, to: float, resolution: float,
                 command: Callable[[str], None] | None = None,
                 width: int = 220, height: int = 34) -> None:
        if not math.isfinite(float(from_)) or not math.isfinite(float(to)):
            raise ValueError("slider range must be finite")
        if float(to) <= float(from_):
            raise ValueError("slider maximum must be greater than minimum")
        if not math.isfinite(float(resolution)) or float(resolution) <= 0:
            raise ValueError("slider resolution must be positive and finite")
        super().__init__(parent, width=width, height=height,
                         highlightthickness=0, borderwidth=0,
                         takefocus=True, background="#F4F1E6")
        self.from_ = float(from_)
        self.to = float(to)
        self.resolution = float(resolution)
        self.command = command
        self._value = self.from_
        self._destroyed = False
        self._bubble_after_id: str | None = None
        self.bind("<Configure>", self._on_configure)
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _decimal_places(self) -> int:
        return max(0, -Decimal(str(self.resolution)).normalize().as_tuple().exponent)

    def _snap(self, value: float) -> float:
        numeric = min(self.to, max(self.from_, float(value)))
        origin = Decimal(str(self.from_))
        step = Decimal(str(self.resolution))
        count = ((Decimal(str(numeric)) - origin) / step).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP)
        snapped = origin + count * step
        clamped = min(Decimal(str(self.to)), max(origin, snapped))
        return float(clamped)

    def _format_value(self, value: float) -> str:
        places = self._decimal_places()
        if places == 0:
            return str(int(round(value)))
        return f"{value:.{places}f}".rstrip("0").rstrip(".")

    def get(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._value = self._snap(value)
        self._redraw()

    def _rail_bounds(self) -> tuple[float, float]:
        width = max(self.winfo_width(), self.winfo_reqwidth())
        return float(self.RAIL_INSET), float(max(self.RAIL_INSET, width - self.RAIL_INSET))

    def _value_to_x(self, value: float) -> float:
        left, right = self._rail_bounds()
        ratio = (self._snap(value) - self.from_) / (self.to - self.from_)
        return left + ratio * (right - left)

    def _x_to_value(self, x: float) -> float:
        left, right = self._rail_bounds()
        if right <= left:
            return self.from_
        ratio = min(1.0, max(0.0, (float(x) - left) / (right - left)))
        return self._snap(self.from_ + ratio * (self.to - self.from_))

    def _redraw(self) -> None:
        pass

    def _on_configure(self, _event=None) -> None:
        self._redraw()

    def _on_destroy(self, _event=None) -> None:
        self._destroyed = True
        if self._bubble_after_id is not None:
            try:
                self.after_cancel(self._bubble_after_id)
            except tk.TclError:
                pass
            self._bubble_after_id = None
```

- [ ] **Step 4: Run component value tests and verify GREEN**

Run:

```powershell
python -m unittest discover -s tests -p test_xp_widgets.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit deterministic value semantics**

```powershell
git add xp_widgets.py tests/test_xp_widgets.py
git commit -m "feat: add XP glossy slider value semantics"
```

---

### Task 2: Add glossy rendering and interactive effects

**Files:**
- Modify: `xp_widgets.py`
- Modify: `tests/test_xp_widgets.py`

**Interfaces:**
- Consumes: Task 1 `XPGlossySlider` value/coordinate helpers.
- Produces: mouse and keyboard bindings, `_set_from_user(value)`, `_show_bubble()`, `_schedule_hide_bubble()`, Canvas tags `rail`, `fill`, `halo`, `thumb`, `focus`, and `bubble`, and safe delayed-bubble teardown.

- [ ] **Step 1: Add failing pointer and keyboard tests**

Append to `tests/test_xp_widgets.py`:

```python
class XPGlossySliderInteractionTests(_SliderTestCase):
    def test_pointer_click_and_drag_emit_snapped_values(self):
        emitted = []
        slider = self.make_slider(command=emitted.append)
        self.root.deiconify()
        self.root.update()
        left, right = slider._rail_bounds()
        slider.event_generate("<Button-1>", x=int((left + right) / 2), y=17)
        slider.event_generate("<B1-Motion>", x=int(right), y=17)
        slider.event_generate("<ButtonRelease-1>", x=int(right), y=17)
        self.root.update()
        self.assertEqual(slider.get(), 100)
        self.assertEqual(emitted[-1], "100")

    def test_arrow_home_and_end_keys_update_and_emit(self):
        emitted = []
        slider = self.make_slider(command=emitted.append)
        self.root.deiconify()
        self.root.update()
        slider.focus_force()
        slider.set(50)
        slider.event_generate("<Right>")
        slider.event_generate("<Up>")
        slider.event_generate("<Home>")
        slider.event_generate("<End>")
        self.root.update()
        self.assertEqual(slider.get(), 100)
        self.assertEqual(emitted, ["51", "52", "0", "100"])
```

- [ ] **Step 2: Add failing Canvas-state and lifecycle tests**

Append:

```python
    def test_hover_press_focus_and_bubble_change_canvas_state(self):
        slider = self.make_slider()
        self.root.deiconify()
        self.root.update()
        slider.event_generate("<Enter>", x=14, y=17)
        self.root.update()
        self.assertTrue(slider.find_withtag("halo"))
        self.assertTrue(slider.find_withtag("bubble"))
        slider.event_generate("<Button-1>", x=14, y=17)
        self.root.update()
        pressed_fill = slider.itemcget("thumb_body", "fill")
        self.assertEqual(pressed_fill, "#356FAF")
        slider.focus_set()
        slider.event_generate("<FocusIn>")
        self.root.update()
        self.assertTrue(slider.find_withtag("focus"))

    def test_destroy_cancels_pending_bubble_hide(self):
        slider = self.make_slider()
        self.root.deiconify()
        self.root.update()
        slider.event_generate("<Enter>", x=14, y=17)
        slider.event_generate("<Leave>", x=14, y=17)
        self.root.update_idletasks()
        self.assertIsNotNone(slider._bubble_after_id)
        slider.destroy()
        self.root.update()
        self.assertTrue(slider._destroyed)
        self.assertIsNone(slider._bubble_after_id)
```

- [ ] **Step 3: Run interaction tests and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_xp_widgets.py -v
```

Expected: Task 1 tests pass; new tests fail because input bindings and tagged rendering do not exist.

- [ ] **Step 4: Implement user input and bubble lifecycle**

Add state initialization and bindings in `XPGlossySlider.__init__()`:

```python
self._hovered = False
self._pressed = False
self._focused = False
self.bind("<Enter>", self._on_enter)
self.bind("<Leave>", self._on_leave)
self.bind("<Button-1>", self._on_press)
self.bind("<B1-Motion>", self._on_drag)
self.bind("<ButtonRelease-1>", self._on_release)
self.bind("<FocusIn>", self._on_focus_in)
self.bind("<FocusOut>", self._on_focus_out)
for sequence, delta in (("<Left>", -1), ("<Down>", -1),
                        ("<Right>", 1), ("<Up>", 1)):
    self.bind(sequence, lambda _event, step=delta: self._on_step(step))
self.bind("<Home>", lambda _event: self._set_from_user(self.from_))
self.bind("<End>", lambda _event: self._set_from_user(self.to))
```

Add methods:

```python
def _set_from_user(self, value: float) -> str:
    self._value = self._snap(value)
    self._show_bubble()
    self._redraw()
    if self.command is not None:
        self.command(self._format_value(self._value))
    return "break"

def _on_enter(self, _event=None) -> None:
    self._hovered = True
    self._show_bubble()
    self._redraw()

def _on_leave(self, _event=None) -> None:
    self._hovered = False
    if not self._pressed:
        self._schedule_hide_bubble()
    self._redraw()

def _on_press(self, event) -> str:
    self.focus_set()
    self._pressed = True
    self.grab_set()
    return self._set_from_user(self._x_to_value(event.x))

def _on_drag(self, event) -> str:
    return self._set_from_user(self._x_to_value(event.x))

def _on_release(self, event) -> str:
    self._pressed = False
    try:
        self.grab_release()
    except tk.TclError:
        pass
    result = self._set_from_user(self._x_to_value(event.x))
    if not self._hovered:
        self._schedule_hide_bubble()
    return result

def _on_step(self, direction: int) -> str:
    return self._set_from_user(self._value + direction * self.resolution)

def _on_focus_in(self, _event=None) -> None:
    self._focused = True
    self._redraw()

def _on_focus_out(self, _event=None) -> None:
    self._focused = False
    self._redraw()

def _show_bubble(self) -> None:
    self._cancel_bubble_hide()

def _cancel_bubble_hide(self) -> None:
    if self._bubble_after_id is None:
        return
    try:
        self.after_cancel(self._bubble_after_id)
    except tk.TclError:
        pass
    self._bubble_after_id = None

def _schedule_hide_bubble(self) -> None:
    self._cancel_bubble_hide()
    self._bubble_after_id = self.after(450, self._hide_bubble)

def _hide_bubble(self) -> None:
    self._bubble_after_id = None
    if self._destroyed:
        return
    self._redraw()
```

Represent bubble visibility with a `_bubble_visible` boolean: set it `True` in
`_show_bubble()`, and `False` in `_hide_bubble()`. Render it whenever
`_bubble_visible or _hovered or _pressed` is true.

- [ ] **Step 5: Implement bounded glossy Canvas rendering**

Replace `_redraw()` with Canvas primitives using these exact colors and tags:

```python
def _redraw(self) -> None:
    if self._destroyed or not self.winfo_exists():
        return
    self.delete("all")
    width = max(self.winfo_width(), self.winfo_reqwidth())
    height = max(self.winfo_height(), self.winfo_reqheight())
    left, right = self._rail_bounds()
    center_y = min(height - 10, 21)
    thumb_x = self._value_to_x(self._value)

    self.create_rectangle(left, center_y - 3, right, center_y + 3,
                          fill="#E5E2D8", outline="#8E9AA6", tags="rail")
    fill_width = max(0.0, thumb_x - left)
    colors = ("#8EB9E8", "#73A7DE", "#5B92CC", "#356FAF",
              "#2C6099", "#244F7D")
    if fill_width > 0:
        segment = fill_width / len(colors)
        for index, color in enumerate(colors):
            x1 = left + index * segment
            x2 = left + (index + 1) * segment
            self.create_rectangle(x1, center_y - 2, x2, center_y + 2,
                                  fill=color, outline=color, tags="fill")

    if self._focused:
        self.create_oval(thumb_x - 11, center_y - 11, thumb_x + 11,
                         center_y + 11, outline="#E4A43A", width=2,
                         tags="focus")
    if self._hovered and not self._pressed:
        self.create_oval(thumb_x - 10, center_y - 10, thumb_x + 10,
                         center_y + 10, fill="#D9EAFB", outline="",
                         tags="halo")

    shadow_y = 2 if not self._pressed else 1
    self.create_oval(thumb_x - 8, center_y - 6 + shadow_y,
                     thumb_x + 8, center_y + 10,
                     fill="#75828E", outline="", tags=("thumb", "thumb_shadow"))
    body_fill = "#356FAF" if self._pressed else "#F7F3E7"
    self.create_oval(thumb_x - 8, center_y - 8, thumb_x + 8, center_y + 8,
                     fill=body_fill, outline="#244F7D", width=1,
                     tags=("thumb", "thumb_body"))
    if not self._pressed:
        self.create_arc(thumb_x - 6, center_y - 6, thumb_x + 6, center_y + 5,
                        start=20, extent=140, style="arc", outline="#FFFFFF",
                        width=2, tags=("thumb", "thumb_highlight"))

    if self._bubble_visible or self._hovered or self._pressed:
        text = self._format_value(self._value)
        bubble_y = max(8, center_y - 18)
        half_width = max(13, 4 + len(text) * 4)
        self.create_rectangle(thumb_x - half_width, bubble_y - 7,
                              thumb_x + half_width, bubble_y + 7,
                              fill="#FFFDF5", outline="#356FAF",
                              tags="bubble")
        self.create_text(thumb_x, bubble_y, text=text, fill="#20252A",
                         font=("Tahoma", 8), tags="bubble")
```

Set `_bubble_visible = False` in the constructor. Update `_on_destroy()` to
call `_cancel_bubble_hide()` before marking the instance destroyed.

- [ ] **Step 6: Run all component tests and verify GREEN**

Run:

```powershell
python -m unittest discover -s tests -p test_xp_widgets.py -v
```

Expected: all 8 component tests pass without Tk callback warnings.

- [ ] **Step 7: Commit rendering and effects**

```powershell
git add xp_widgets.py tests/test_xp_widgets.py
git commit -m "feat: render interactive XP glossy sliders"
```

---

### Task 3: Integrate glossy sliders throughout Jitter

**Files:**
- Modify: `ui.py:1-15,354-377`
- Modify: `tests/test_ui.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `XPGlossySlider` constructor, `set(float)`, `get()`, and string-valued user `command` callback.
- Produces: every existing `self.<motion_key>_scale` attribute as an `XPGlossySlider`; unchanged `_scale_changed(key: str, value: str)` integration.

- [ ] **Step 1: Add a failing all-controls adoption test**

Add `from xp_widgets import XPGlossySlider` to `tests/test_ui.py`, then add:

```python
def test_every_numeric_control_uses_xp_glossy_slider(self):
    numeric_keys = (
        "motion_angle_deg", "motion_strength_pps", "horizontal_jitter_pps",
        "vertical_jitter_pps", "jitter_rate_hz",
        "jitter_randomness_percent", "jitter_axis_phase_deg",
        "smoothness_percent", "ramp_up_ms", "update_rate_hz",
        "max_step_px", "acceleration_pps2", "deceleration_pps2",
    )
    for key in numeric_keys:
        with self.subTest(key=key):
            self.assertIsInstance(getattr(self.app, f"{key}_scale"),
                                  XPGlossySlider)
```

- [ ] **Step 2: Add failing synchronization tests**

Add:

```python
def test_glossy_slider_user_change_updates_exact_entry_and_snapshot(self):
    slider = self.app.motion_strength_pps_scale
    slider._set_from_user(123)
    self.app.update()
    self.assertEqual(self.app.motion_strength_pps_var.get(), "123")
    self.assertEqual(self.app.get_motion_settings().strength_pps, 123.0)

def test_exact_entry_and_preset_changes_update_glossy_slider_silently(self):
    slider = self.app.motion_strength_pps_scale
    self.app.motion_strength_pps_var.set("77")
    self.app.update()
    self.assertEqual(slider.get(), 77.0)
    self.app.preset_var.set("Balanced")
    self.app.apply_preset()
    self.app.update()
    self.assertEqual(slider.get(),
                     self.app.get_motion_settings().strength_pps)
```

- [ ] **Step 3: Run focused UI tests and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_ui.py -k "glossy_slider" -v
```

Expected: adoption test fails because existing controls are `tk.Scale`; user-change test errors because `_set_from_user` does not exist.

- [ ] **Step 4: Replace only the shared scale construction**

In `ui.py`, import the component:

```python
from xp_widgets import XPGlossySlider
```

Replace the `tk.Scale(...)` block inside `_numeric_control()` with:

```python
slider = XPGlossySlider(
    block,
    from_=low,
    to=high,
    resolution=resolution,
    command=lambda value, name=key: self._scale_changed(name, value),
)
slider.set(float(self.motion_vars[key].get()))
slider.pack(fill="x", pady=(2, 0))
```

Keep the existing entry creation, public attribute assignments, variable
traces, `_scale_changed()`, and `_motion_changed()` unchanged. In
`apply_preset()`, update each numeric slider silently alongside its variable so
preset changes visibly synchronize even while `_updating_motion_controls` blocks
trace callbacks:

```python
for key, value in motion_settings_to_mapping(settings).items():
    variable = self.motion_vars[key]
    variable.set(bool(value) if key == "jitter_enabled" else str(value))
    scale = getattr(self, f"{key}_scale", None)
    if scale is not None:
        scale.set(float(value))
```

- [ ] **Step 5: Run focused UI and component tests**

Run:

```powershell
python -m unittest discover -s tests -p test_xp_widgets.py -v
python -m unittest discover -s tests -p test_ui.py -v
```

Expected: component tests and all UI tests pass; no Tk callback warnings.

- [ ] **Step 6: Run the current complete suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: every hardware-free test passes.

- [ ] **Step 7: Commit UI integration**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: use XP glossy sliders across Jitter"
```

Before committing, inspect `git diff -- ui.py tests/test_ui.py` and preserve all
pre-existing uncommitted XP Remastered button, banner-removal, and native-title
changes rather than reverting them.

---

### Task 4: Verify the complete application and visual effects

**Files:**
- Verify: `xp_widgets.py`
- Verify: `ui.py`
- Test: `tests/test_xp_widgets.py`
- Test: `tests/test_ui.py`
- Test: `tests/`

**Interfaces:**
- Consumes: completed `XPGlossySlider` integration and the existing `main.py` entry point.
- Produces: fresh syntax, automated, import, and Windows visual smoke evidence.

- [ ] **Step 1: Compile every application module including the new component**

Run:

```powershell
python -m py_compile main.py ui.py xp_widgets.py motion.py makcu_service.py hotkeys.py settings.py
```

Expected: exit code 0 with no output.

- [ ] **Step 2: Run the complete automated suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass without leaked windows or Tk callback warnings.

- [ ] **Step 3: Verify Makcu import**

Run:

```powershell
python -c "import makcu"
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Perform Windows source UI smoke checks**

Run `python main.py` and verify:

- every Quick Jitter and Advanced numeric control uses the glossy rail/thumb;
- hover shows a blue halo and readable transient value bubble;
- pressing and dragging visually depresses the thumb and updates the exact entry;
- arrow keys and Home/End update both slider and exact entry;
- typing a valid exact value and applying a preset update the thumb silently;
- expanding and scrolling Advanced keeps the `640x560` outer geometry;
- STOP remains completely visible at both ends of the scroll range;
- closing the window exits cleanly.

Do not invoke Test 3s or normal motion merely to inspect sliders. Hardware
behavior is unchanged and is not required for this presentation-only smoke run.

- [ ] **Step 5: Review final state and commit verification corrections only if needed**

Run:

```powershell
git diff --check
git status --short
git diff -- xp_widgets.py ui.py tests/test_xp_widgets.py tests/test_ui.py
```

If a correction was necessary, rerun Steps 1-3 and commit only relevant files:

```powershell
git add xp_widgets.py ui.py tests/test_xp_widgets.py tests/test_ui.py
git commit -m "fix: polish XP glossy slider behavior"
```
