# Liquid Control Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active Windows XP presentation with a fixed-size, dual-theme Liquid Control Deck while preserving every runtime and safety behavior.

**Architecture:** Rename the isolated Canvas widget module and evolve its navigation and slider contracts before integrating a new icon button. Rebuild only the Tkinter presentation hierarchy in `ui.py`; continue routing all settings, worker events, motion, Makcu calls, and shutdown through the existing application methods.

**Tech Stack:** Python 3, standard-library Tkinter/ttk, `unittest`, existing `makcu` runtime dependency

**Spec:** `docs/superpowers/specs/2026-08-24-liquid-control-deck-design.md`

## Global Constraints

- The interface is an English-only, fixed-size Windows Tkinter application approximately `780x640`.
- Support complete `dark` and `light` themes without raster assets or new runtime dependencies.
- Keep primary cyan text dark and secondary controls readable in every state.
- Keep connection colors green, amber, and red.
- Keep all Tk widget and Tk-variable access on the main thread.
- Keep `motion.py`, `makcu_service.py`, `hotkeys.py`, and the settings schema behavior unchanged.
- Keep STOP persistent and immediately interruptible; keep Test 3s on the production motion engine.
- Keep Advanced content internally scrollable without resizing the outer window.
- Track and cancel every new Tk callback during teardown.
- Historical design and plan documents remain unchanged.
- Follow TDD: observe each targeted test fail before production implementation.

## File Map

- Create `liquid_widgets.py`: presentation-only `LiquidNavigation`, `LiquidSlider`, and `LiquidIconButton` Canvas widgets.
- Delete `xp_widgets.py`: superseded widget module after imports migrate.
- Create `tests/test_liquid_widgets.py`: unit tests for all custom liquid widgets.
- Delete `tests/test_xp_widgets.py`: superseded XP-named tests after equivalent behavior migrates.
- Modify `ui.py`: liquid palettes/styles, shell hierarchy, page allocation, widget integration, theme propagation, and teardown.
- Modify `tests/test_ui.py`: liquid layout/style acceptance tests while preserving runtime regression tests.

---

### Task 1: Rename the Widget Module and Public Contracts

**Files:**
- Create: `liquid_widgets.py`
- Create: `tests/test_liquid_widgets.py`
- Modify: `ui.py:28`
- Modify: `tests/test_ui.py:9`
- Delete: `xp_widgets.py`
- Delete: `tests/test_xp_widgets.py`

**Interfaces:**
- Consumes: existing `LiquidXPNav` and `XPGlossySlider` behavior.
- Produces: `LiquidNavigation(tk.Canvas)` and `LiquidSlider(tk.Canvas)` with the existing constructor, `select`, `set_palette`, `cancel_animation`, `get`, and `set` semantics.

- [ ] **Step 1: Create the renamed failing test contract**

Move the XP widget tests into `tests/test_liquid_widgets.py`, import the new names, and rename test classes:

```python
from liquid_widgets import LiquidNavigation, LiquidSlider

class LiquidSliderValueTests(_SliderTestCase):
    slider_type = LiquidSlider

class LiquidSliderInteractionTests(_SliderTestCase):
    slider_type = LiquidSlider

class LiquidNavigationTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.nav = LiquidNavigation(
            self.root, labels=("Control", "Motion", "Advanced"),
            command=lambda index: self.selected.append(index),
        )
```

Update `_SliderTestCase.make_slider()` to construct `self.slider_type`, and preserve all current value, interaction, animation, scheduler-failure, rounded-shape, and destruction assertions.

- [ ] **Step 2: Run the renamed tests and verify RED**

Run: `python -m unittest discover -s tests -p test_liquid_widgets.py -v`

Expected: FAIL because `liquid_widgets` does not exist.

- [ ] **Step 3: Rename the module and classes minimally**

Move the implementation into `liquid_widgets.py` and rename:

```python
class LiquidNavigation(tk.Canvas):
    """Keyboard-accessible animated navigation for the liquid dashboard."""

class LiquidSlider(tk.Canvas):
    """Exact-value Canvas slider for the liquid dashboard."""
```

Rename `LIGHT_NAV_PALETTE` to `DEFAULT_NAV_PALETTE` and
`LIGHT_SLIDER_PALETTE` to `DEFAULT_SLIDER_PALETTE`. Update imports and
`isinstance` checks in `ui.py` and `tests/test_ui.py`. Change the navigation
labels to `("Control", "Motion", "Advanced")`. Do not change rendering yet.

- [ ] **Step 4: Run widget and UI tests**

Run: `python -m unittest discover -s tests -p test_liquid_widgets.py -v`

Expected: PASS.

Run: `python -m unittest discover -s tests -p test_ui.py -v`

Expected: PASS after import/name expectation updates; runtime behavior remains unchanged.

- [ ] **Step 5: Commit the rename**

```powershell
git add liquid_widgets.py tests/test_liquid_widgets.py ui.py tests/test_ui.py
git add -u xp_widgets.py tests/test_xp_widgets.py
git commit -m "refactor: rename liquid widget contracts"
```

### Task 2: Implement the Liquid Visual Palettes and Rendering

**Files:**
- Modify: `liquid_widgets.py`
- Modify: `tests/test_liquid_widgets.py`

**Interfaces:**
- Consumes: Task 1 `LiquidNavigation` and `LiquidSlider` APIs.
- Produces: `DEFAULT_NAV_PALETTE`, `DEFAULT_SLIDER_PALETTE`, rounded liquid rendering, and palette-driven focus/disabled states.

- [ ] **Step 1: Write failing liquid-rendering tests**

Add assertions that use semantic Canvas tags instead of pixel screenshots:

```python
def test_navigation_palette_updates_glass_lens_and_focus_ring(self):
    palette = {
        "background": "#111827", "surface": "#1B2638",
        "surface_highlight": "#33445E", "border": "#45566F",
        "lens": "#63E6FF", "lens_highlight": "#B8F6FF",
        "text": "#EDF7FF", "selected_text": "#08212A",
        "focus": "#FFE08A",
    }
    self.nav.set_palette(palette)
    self.assertEqual(self.nav.itemcget("glass", "fill"), "#1B2638")
    self.assertEqual(self.nav.itemcget("lens", "fill"), "#63E6FF")

def test_slider_palette_updates_rail_fill_thumb_and_disabled_colors(self):
    slider = self.make_slider()
    slider.set_palette({
        "background": "#111827", "rail": "#27364A", "fill": "#63E6FF",
        "thumb": "#E8FBFF", "thumb_border": "#63E6FF",
        "halo": "#315F70", "text": "#EDF7FF", "bubble": "#24384A",
        "bubble_text": "#EDF7FF", "focus": "#FFE08A",
        "disabled": "#536174", "disabled_text": "#8C99AA",
    })
    self.assertEqual(slider.itemcget("rail", "fill"), "#27364A")
    self.assertEqual(slider.itemcget("fill", "fill"), "#63E6FF")
```

Also assert there are no `XP` names in exported classes or palette constants.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m unittest discover -s tests -p test_liquid_widgets.py -v`

Expected: FAIL because the semantic palette keys and Canvas tags are absent.

- [ ] **Step 3: Implement deterministic liquid rendering**

Replace bevel/gloss tags with layered rounded Canvas shapes:

```python
DEFAULT_NAV_PALETTE = {
    "background": "#F2F7FA", "surface": "#E5F0F5",
    "surface_highlight": "#FFFFFF", "border": "#B9CBD5",
    "lens": "#55DDF6", "lens_highlight": "#C7F8FF",
    "text": "#263640", "selected_text": "#07252C", "focus": "#8B5CF6",
}

DEFAULT_SLIDER_PALETTE = {
    "background": "#F2F7FA", "rail": "#C9D9E1", "fill": "#55DDF6",
    "thumb": "#F8FEFF", "thumb_border": "#33BDD8", "halo": "#B7EFF8",
    "text": "#263640", "bubble": "#244653", "bubble_text": "#FFFFFF",
    "focus": "#8B5CF6", "disabled": "#A9B6BC", "disabled_text": "#7A878D",
}
```

Keep current interpolation, clamping, snapping, callback, and scheduling logic.
Draw navigation layers with `glass`, `glass-highlight`, `lens`, and
`lens-highlight` tags. Draw sliders with `rail`, `fill`, `halo`, `thumb`, and
`focus-ring` tags. Never rely on actual alpha transparency.

- [ ] **Step 4: Run the widget suite**

Run: `python -m unittest discover -s tests -p test_liquid_widgets.py -v`

Expected: PASS with all prior semantic and shutdown tests retained.

- [ ] **Step 5: Commit liquid rendering**

```powershell
git add liquid_widgets.py tests/test_liquid_widgets.py
git commit -m "feat: render liquid navigation and sliders"
```

### Task 3: Add the Mini Liquid Icon Button

**Files:**
- Modify: `liquid_widgets.py`
- Modify: `tests/test_liquid_widgets.py`

**Interfaces:**
- Consumes: palette conventions from Task 2.
- Produces: `LiquidIconButton(parent, *, icon: str, accessible_name: str, command: Callable[[], None], palette: Mapping[str, str] | None = None, size: int = 34)`, `set_palette`, `set_enabled`, and `accessible_name`.

- [ ] **Step 1: Write failing icon-button tests**

```python
class LiquidIconButtonTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.calls = []
        self.button = LiquidIconButton(
            self.root, icon="↻", accessible_name="Reconnect Makcu",
            command=lambda: self.calls.append("called"),
        )

    def test_click_enter_and_space_activate_when_enabled(self):
        self.button._activate()
        self.button.event_generate("<Return>")
        self.button.event_generate("<space>")
        self.root.update()
        self.assertEqual(self.calls, ["called", "called", "called"])

    def test_disabled_button_does_not_activate(self):
        self.button.set_enabled(False)
        self.button._activate()
        self.assertEqual(self.calls, [])

    def test_palette_redraws_surface_icon_and_focus(self):
        self.button.set_palette(DARK_ICON_PALETTE)
        self.assertEqual(self.button.itemcget("surface", "fill"), "#243247")
        self.assertEqual(self.button.itemcget("icon", "fill"), "#EAF7FF")
```

Include destroy and focus tests that confirm no scheduled callback survives.

- [ ] **Step 2: Run icon tests and verify RED**

Run: `python -m unittest discover -s tests -p test_liquid_widgets.py -k LiquidIconButton -v`

Expected: FAIL because `LiquidIconButton` is not defined.

- [ ] **Step 3: Implement the button**

Implement a fixed-size, focusable Canvas. Bind pointer enter/leave/press/release,
`<Return>`, and `<space>`. `_activate()` invokes `command` only when enabled.
`set_palette()` replaces a complete palette and redraws. `set_enabled()` updates
state and redraws. Store `accessible_name` publicly for tooltip and tests.

- [ ] **Step 4: Run all liquid widget tests**

Run: `python -m unittest discover -s tests -p test_liquid_widgets.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the icon component**

```powershell
git add liquid_widgets.py tests/test_liquid_widgets.py
git commit -m "feat: add liquid mini icon buttons"
```

### Task 4: Rebuild the Shell as a Liquid Control Deck

**Files:**
- Modify: `ui.py:31-789`
- Modify: `tests/test_ui.py:82-548`

**Interfaces:**
- Consumes: Task 2 `LiquidNavigation`, `LiquidSlider`; Task 3 `LiquidIconButton`.
- Produces: `identity_frame`, `navigation_frame`, `page_host`, `control_page`, `motion_page`, `advanced_page`, `runtime_frame`, `footer_frame`, and unchanged action attributes.

- [ ] **Step 1: Replace XP acceptance tests with failing liquid shell tests**

Update layout tests to require:

```python
def test_window_is_fixed_size_liquid_control_deck(self):
    self.assertEqual(self.app.geometry().split("+")[0], "780x640")
    self.assertFalse(self.app.resizable()[0])
    self.assertFalse(self.app.resizable()[1])

def test_shell_region_order_is_identity_nav_page_runtime_footer(self):
    widgets = (self.app.identity_frame, self.app.navigation_frame,
               self.app.page_host, self.app.runtime_frame,
               self.app.footer_frame)
    self.assertEqual([int(widget.grid_info()["row"]) for widget in widgets],
                     [0, 1, 2, 3, 4])

def test_navigation_owns_control_motion_and_advanced_pages(self):
    self.assertEqual(self.app.nav.labels, ("Control", "Motion", "Advanced"))
    self.assertEqual(self.app.pages,
                     (self.app.control_page, self.app.motion_page,
                      self.app.advanced_page))

def test_mini_actions_are_liquid_icon_buttons(self):
    for button in (self.app.reconnect_button, self.app.test_button,
                   self.app.theme_button):
        self.assertIsInstance(button, LiquidIconButton)
        self.assertIs(button.master, self.app.navigation_actions)
```

Assert Control owns Trigger, Modifier, Preset, Hotkey, and device summary;
Motion owns Strength and Rate; Advanced owns waveform, curve, and remaining
numeric controls. Preserve tests for one visible page, persistent STOP,
scrolling, geometry stability, tooltips, and runtime operations.

- [ ] **Step 2: Run UI tests and verify RED**

Run: `python -m unittest discover -s tests -p test_ui.py -v`

Expected: FAIL for geometry, region names, page labels, component types, allocation, and Liquid style expectations.

- [ ] **Step 3: Define liquid palettes and ttk styles**

Replace all active `XP_*`, `XP.*`, and Tahoma declarations with semantic
Liquid constants and Segoe UI:

```python
FONT_FAMILY = "Segoe UI"
BODY_FONT = (FONT_FAMILY, 10)
SMALL_FONT = (FONT_FAMILY, 9)
TITLE_FONT = (FONT_FAMILY, 18, "bold")
SECTION_FONT = (FONT_FAMILY, 9, "bold")

DARK_PALETTE = {
    "window": "#0D1420", "surface": "#172232", "raised": "#202F43",
    "border": "#34465C", "text": "#EEF8FF", "muted": "#91A5B8",
    "accent": "#63E6FF", "accent_hover": "#8CECFF",
    "accent_pressed": "#3CC7E1", "green": "#42D392",
    "amber": "#F6C85F", "red": "#FF6B78", "danger": "#D64052",
    "focus": "#FFE08A",
}
```

Add a complete matching light palette. Register only `Liquid.*` ttk styles:
`Liquid.App.TFrame`, `Liquid.Surface.TFrame`, `Liquid.Title.TLabel`,
`Liquid.Body.TLabel`, `Liquid.Muted.TLabel`, `Liquid.Primary.TButton`,
`Liquid.Secondary.TButton`, `Liquid.Danger.TButton`, connection-state labels,
entries, invalid entries, and readonly comboboxes.

- [ ] **Step 4: Build the five-region hierarchy and allocate pages**

Set `self.geometry("780x640")` and `self.resizable(False, False)`. Rebuild
`_build_page()` in the exact region order. Rename `setup_page` to
`control_page`. Mount device/Trigger/Modifier/Preset/Hotkey under Control,
Strength/Rate under Motion, and the existing scrollable workspace under
Advanced. Keep `self.pages` persistent and change `select_page()` only enough
to address the renamed page.

Build Reconnect (`↻`), Test 3s (`▶`), and theme (`☀`/`☾`) with
`LiquidIconButton`. Preserve the existing public button attributes,
`reconnect_tooltip_text`, `test_tooltip_text`, and `theme_tooltip_text` so
runtime and tooltip tests remain stable.

- [ ] **Step 5: Integrate sliders and preserve runtime dock behavior**

Construct `LiquidSlider` in `_numeric_control()`. Keep exact entry binding,
silent `set()` synchronization, preset application, validation, and motion
snapshot replacement unchanged. Keep Enable, state, and STOP in
`runtime_frame`, outside every page and scrollable container.

- [ ] **Step 6: Run UI and complete tests**

Run: `python -m unittest discover -s tests -p test_ui.py -v`

Expected: PASS.

Run: `python -m unittest discover -s tests -v`

Expected: PASS with all service, motion, settings, entrypoint, and hotkey tests unchanged.

- [ ] **Step 7: Commit the shell**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: build Liquid Control Deck shell"
```

### Task 5: Complete Theme Propagation, Accessibility, and Shutdown Safety

**Files:**
- Modify: `ui.py`
- Modify: `liquid_widgets.py`
- Modify: `tests/test_ui.py`
- Modify: `tests/test_liquid_widgets.py`

**Interfaces:**
- Consumes: completed liquid shell and widget APIs.
- Produces: `_navigation_palette()`, `_slider_palette()`, `_icon_palette()`, complete live theme refresh, and cancellation-safe teardown.

- [ ] **Step 1: Write failing integration and teardown tests**

```python
def test_theme_toggle_updates_every_custom_liquid_widget(self):
    self.app.toggle_theme()
    self.assertEqual(self.app.nav.cget("background"), "#0D1420")
    slider_keys = (
        "motion_strength_pps", "jitter_rate_hz", "motion_angle_deg",
        "horizontal_jitter_pps", "vertical_jitter_pps",
        "jitter_randomness_percent", "jitter_axis_phase_deg",
        "smoothness_percent", "ramp_up_ms", "update_rate_hz",
        "max_step_px", "acceleration_pps2", "deceleration_pps2",
    )
    for key in slider_keys:
        slider = getattr(self.app, f"{key}_scale")
        self.assertEqual(slider.cget("background"), "#0D1420")
    for button in (self.app.reconnect_button, self.app.test_button,
                   self.app.theme_button):
        self.assertEqual(button.cget("background"), "#0D1420")

def test_close_cancels_all_custom_widget_callbacks(self):
    self.app.nav.select(2)
    self.app.close_app()
    self.assertIsNone(self.app.nav._animation_after_id)
```

Add assertions for readable focus/disabled colors, navigation Home/End keys,
icon accessible names, icon keyboard activation, and Test icon disabled only
during an active Test Run.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m unittest discover -s tests -p test_ui.py -k theme -v`

Expected: FAIL because one or more custom widget palettes are stale.

Run: `python -m unittest discover -s tests -p test_liquid_widgets.py -v`

Expected: FAIL for missing accessibility or teardown expectations.

- [ ] **Step 3: Implement complete palette propagation**

Return complete dictionaries from `_navigation_palette()`,
`_slider_palette()`, and `_icon_palette()`. In `toggle_theme()`, reconfigure
ttk styles and call `set_palette()` for navigation, every slider, and all icon
buttons. Change the theme icon and accessible tooltip text after the palette
update. Preserve persisted theme values `dark` and `light`.

- [ ] **Step 4: Finish keyboard and teardown behavior**

Bind Home and End in `LiquidNavigation`. Ensure every custom widget cancels its
own callback from its destroy handler. In `close_app()`, set `_closing` first,
cancel navigation animation and transient widget callbacks, then continue the
existing queue, hotkey, movement, service, save, and Tk destruction sequence.
Catch only expected `tk.TclError` scheduler teardown failures.

- [ ] **Step 5: Run the complete suite**

Run: `python -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 6: Commit accessibility and safety polish**

```powershell
git add ui.py liquid_widgets.py tests/test_ui.py tests/test_liquid_widgets.py
git commit -m "fix: complete liquid themes and teardown safety"
```

### Task 6: Remove Active XP Naming and Perform Final Verification

**Files:**
- Modify: `README.md` only if it describes the active XP interface
- Modify: `ui.py`
- Modify: `liquid_widgets.py`
- Modify: `tests/test_ui.py`
- Modify: `tests/test_liquid_widgets.py`

**Interfaces:**
- Consumes: Tasks 1-5 completed implementation.
- Produces: a clean Liquid Control Deck with no active XP source naming and verified repository behavior.

- [ ] **Step 1: Scan active source for XP remnants**

Run:

```powershell
rg -n "XP|Luna|xp_widgets|LiquidXPNav|XPGlossySlider" ui.py liquid_widgets.py tests README.md
```

Expected: no matches in active UI source or tests. References inside historical
`docs/superpowers/` files are intentionally retained.

- [ ] **Step 2: Remove any reported active remnants**

Rename remaining active identifiers to semantic Liquid names and update their
tests in the same change. Do not edit historical specifications or plans.

- [ ] **Step 3: Run repository-required verification**

Run:

```powershell
python -m py_compile main.py ui.py motion.py makcu_service.py hotkeys.py settings.py
python -m unittest discover -s tests -v
python -c "import makcu"
git diff --check
```

Expected: every command exits `0` and the complete unit suite reports `OK`.

- [ ] **Step 4: Perform a source UI smoke check without moving hardware**

Run `python main.py`, inspect both themes and all three pages, confirm STOP and
the mini actions remain visible, scroll Advanced, switch back to Control, and
close the window. Do not invoke Enable or Test 3s without intentional hardware
verification. Confirm closing returns promptly.

- [ ] **Step 5: Commit final source polish**

```powershell
git add ui.py liquid_widgets.py tests/test_ui.py tests/test_liquid_widgets.py README.md
git commit -m "chore: finalize Liquid Control Deck redesign"
```

- [ ] **Step 6: Record the hardware verification boundary**

Report that connection, Trigger/Modifier movement, Reconnect, Test 3s, global
hotkey, STOP, disconnect, and shutdown still require a connected Makcu device.
Do not run Nuitka or `gen.bat` unless the user explicitly requests packaging.
