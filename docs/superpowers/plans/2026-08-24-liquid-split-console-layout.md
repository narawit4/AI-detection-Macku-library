# Liquid Split Console Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rearrange the existing Liquid Control Deck into a fixed `840x620` split console with a persistent vertical navigation rail and page-specific asymmetric workspace.

**Architecture:** Extend the presentation-only `LiquidNavigation` with a backward-compatible orientation contract, then rebuild only the widget hierarchy and geometry in `ui.py`. Preserve all existing variables, action attributes, motion snapshots, service callbacks, safety paths, theme roles, and custom-widget lifecycle behavior.

**Tech Stack:** Python 3, standard-library Tkinter/ttk, `unittest`, existing `makcu` dependency

**Spec:** `docs/superpowers/specs/2026-08-24-liquid-split-console-layout-design.md`

## Global Constraints

- Fixed `840x620` English-only Windows Tkinter window.
- Persistent `176` pixel left rail and flexible right workspace.
- Preserve both themes, text contrast `>=4.5:1`, and icon contrast `>=3:1`.
- Preserve `Control`, `Motion`, and `Advanced` as persistent pages.
- Keep Reconnect, Test 3s, theme action, connection status, footer, runtime dock, Enable, and STOP persistent.
- Keep Advanced as the only scrollable page.
- Do not change motion, Makcu, hotkey, settings schema, Test Run, STOP, or bounded shutdown behavior.
- Add no dependencies, raster assets, hardware movement, tray, or packaging work.
- Keep all Tk access on the main thread and cancel every scheduled callback before teardown.
- Follow TDD and commit each independently reviewed task.

## File Map

- Modify `liquid_widgets.py`: orientation-aware navigation geometry/input/rendering.
- Modify `tests/test_liquid_widgets.py`: vertical navigation contract and horizontal regression coverage.
- Modify `ui.py`: split shell, persistent rail, page layouts, footer, and runtime dock geometry.
- Modify `tests/test_ui.py`: layout ownership, geometry, visibility, theme, and runtime regression tests.
- Modify `README.md`: update active layout wording only if it describes the previous top-navigation hierarchy.

---

### Task 1: Add Vertical Liquid Navigation

**Files:**
- Modify: `liquid_widgets.py:37-352`
- Modify: `tests/test_liquid_widgets.py`

**Interfaces:**
- Consumes: existing `LiquidNavigation(..., labels, command, palette=None)` behavior.
- Produces: `LiquidNavigation(..., orientation: str = "horizontal")`, accepting exactly `horizontal` or `vertical`.

- [ ] **Step 1: Write failing orientation tests**

Add a vertical fixture and tests:

```python
def make_vertical_nav(self):
    return LiquidNavigation(
        self.root,
        labels=("Control", "Motion", "Advanced"),
        command=self.selected.append,
        orientation="vertical",
        width=152,
        height=216,
    )

def test_rejects_unknown_orientation(self):
    with self.assertRaisesRegex(ValueError, "orientation"):
        LiquidNavigation(self.root, labels=("A",), command=lambda _i: None,
                         orientation="diagonal")

def test_vertical_navigation_stacks_destinations_and_moves_lens_on_y_axis(self):
    nav = self.make_vertical_nav()
    nav.pack()
    self.root.update()
    first = nav._target_lens_position(0)
    last = nav._target_lens_position(2)
    self.assertEqual(first[0], last[0])
    self.assertLess(first[1], last[1])

def test_vertical_up_down_home_end_and_activation(self):
    nav = self.make_vertical_nav()
    nav.select(1, animate=False)
    nav.event_generate("<Down>")
    nav.event_generate("<Up>")
    nav.event_generate("<End>")
    nav.event_generate("<Return>")
    self.root.update()
    self.assertEqual(nav.selected_index, 2)
    self.assertEqual(self.selected[-1], 2)
```

Retain the existing horizontal Left/Right/Home/End/Enter/Space tests unchanged.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m unittest discover -s tests -p test_liquid_widgets.py -k LiquidNavigation -v`

Expected: FAIL because `orientation` and vertical geometry do not exist.

- [ ] **Step 3: Implement orientation-aware geometry and input**

Validate and store orientation:

```python
if orientation not in {"horizontal", "vertical"}:
    raise ValueError("orientation must be 'horizontal' or 'vertical'")
self.orientation = orientation
```

Replace axis-specific helpers with `_item_bounds(index) -> tuple[float, float,
float, float]` and `_target_lens_position(index) -> tuple[float, float]`.
Horizontal mode retains existing x interpolation. Vertical mode interpolates y,
stacks labels evenly, maps Up/Down to adjacent selection, and maps pointer y to
the hit destination. Preserve animation generation, scheduler-error boundaries,
focus rendering, activation, palette updates, and destroy cancellation.

- [ ] **Step 4: Run widget and full regression tests**

Run: `python -m unittest discover -s tests -p test_liquid_widgets.py -v`

Expected: PASS.

Run: `python -m unittest discover -s tests -v`

Expected: PASS with existing horizontal UI unchanged.

- [ ] **Step 5: Commit**

```powershell
git add liquid_widgets.py tests/test_liquid_widgets.py
git commit -m "feat: add vertical liquid navigation"
```

### Task 2: Build the Split Shell and Persistent Rail

**Files:**
- Modify: `ui.py:84-875`
- Modify: `tests/test_ui.py:80-760`

**Interfaces:**
- Consumes: Task 1 vertical `LiquidNavigation` and existing `LiquidIconButton`.
- Produces: `shell`, `navigation_rail`, `rail_identity`, `connection_indicator`, `nav`, `navigation_actions`, `console_workspace`, `page_host`, `footer_frame`, and `runtime_frame`.

- [ ] **Step 1: Write failing split-shell tests**

```python
def test_window_is_fixed_size_liquid_split_console(self):
    self.assertEqual(self.app.geometry().split("+")[0], "840x620")
    self.assertEqual(self.app.resizable(), (False, False))

def test_shell_uses_persistent_rail_and_console_columns(self):
    self.assertEqual(int(self.app.navigation_rail.grid_info()["column"]), 0)
    self.assertEqual(int(self.app.console_workspace.grid_info()["column"]), 1)
    self.assertEqual(int(self.app.navigation_rail.cget("width")), 176)
    self.assertEqual(self.app.nav.orientation, "vertical")

def test_rail_owns_identity_connection_navigation_and_mini_actions(self):
    for widget in (self.app.rail_identity, self.app.connection_indicator,
                   self.app.nav, self.app.navigation_actions):
        self.assertTrue(self._is_descendant(widget, self.app.navigation_rail))
    for button in (self.app.reconnect_button, self.app.test_button,
                   self.app.theme_button):
        self.assertIs(button.master, self.app.navigation_actions)

def test_workspace_keeps_page_footer_runtime_order(self):
    widgets = (self.app.page_host, self.app.footer_frame, self.app.runtime_frame)
    self.assertEqual([int(widget.grid_info()["row"]) for widget in widgets],
                     [0, 1, 2])
```

Update existing persistent-page, STOP visibility, page geometry, tooltip,
connection-indicator, shell Canvas-tag, and theme tests to address the new
parents without weakening their behavior assertions.

- [ ] **Step 2: Run UI tests and verify RED**

Run: `python -m unittest discover -s tests -p test_ui.py -v`

Expected: FAIL for geometry, missing rail/workspace, orientation, and ownership.

- [ ] **Step 3: Rebuild the two-column shell**

Set `840x620`. Keep `shell` as the theme-aware Canvas and grid two direct
children: `navigation_rail` at column 0 with fixed width 176, and
`console_workspace` at column 1 with weight 1. Draw semantic `rail-surface`,
`workspace-band`, `floating-panel`, and `panel-highlight` layers behind them.

Move identity, subtitle, connection glow/text, vertical nav, and mini actions
into the rail. Anchor actions at its bottom. Move `page_host`, footer, and
runtime dock into console rows 0, 1, 2. Preserve every public action attribute,
tooltip string, page tuple, and runtime method.

- [ ] **Step 4: Preserve complete theme redraw and connection state**

Update shell/rail redraw coordinates and palette propagation without creating
callbacks. Keep semantic tags for both theme tests. Continue updating
`status-glow`, `status-marker`, and state tags for connecting/connected/
disconnected. Do not duplicate the device service state.

- [ ] **Step 5: Run UI and full suites**

Run: `python -m unittest discover -s tests -p test_ui.py -v`

Expected: PASS.

Run: `python -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: build Liquid Split Console shell"
```

### Task 3: Arrange Control, Motion, and Advanced Workspaces

**Files:**
- Modify: `ui.py`
- Modify: `tests/test_ui.py`

**Interfaces:**
- Consumes: Task 2 `navigation_rail`, `console_workspace`, and persistent pages.
- Produces: `control_bindings_card`, `control_device_card`, `motion_hero_card`, and `motion_summary_card`; existing Advanced scroll components remain page-local.

- [ ] **Step 1: Write failing ownership and geometry tests**

```python
def test_control_uses_asymmetric_bindings_and_device_columns(self):
    self.assertEqual(int(self.app.control_bindings_card.grid_info()["column"]), 0)
    self.assertEqual(int(self.app.control_device_card.grid_info()["column"]), 1)
    self.assertGreater(
        int(self.app.control_page.grid_columnconfigure(0)["weight"]),
        int(self.app.control_page.grid_columnconfigure(1)["weight"]),
    )
    for widget in (self.app.trigger_combo, self.app.modifier_combo,
                   self.app.hotkey_button):
        self.assertTrue(self._is_descendant(widget,
                                            self.app.control_bindings_card))
    self.assertTrue(self._is_descendant(self.app.preset_combo,
                                        self.app.control_device_card))

def test_motion_uses_hero_controls_and_snapshot_side_card(self):
    self.assertEqual(int(self.app.motion_hero_card.grid_info()["column"]), 0)
    self.assertEqual(int(self.app.motion_summary_card.grid_info()["column"]), 1)
    for key in ("motion_strength_pps", "jitter_rate_hz"):
        self.assertTrue(self._is_descendant(getattr(self.app, f"{key}_scale"),
                                            self.app.motion_hero_card))
    self.assertTrue(self._is_descendant(self.app.motion_summary_label,
                                        self.app.motion_summary_card))

def test_advanced_remains_the_only_scrollable_page(self):
    self.assertTrue(self._is_descendant(self.app.advanced_canvas,
                                        self.app.advanced_page))
    self.assertFalse(self._is_descendant(self.app.footer_frame,
                                         self.app.advanced_canvas))
```

Retain snapshot synchronization tests for valid edits, presets, and invalid
edits.

- [ ] **Step 2: Run focused layout tests and verify RED**

Run: `python -m unittest discover -s tests -p test_ui.py -k asymmetric -v`

Run: `python -m unittest discover -s tests -p test_ui.py -k hero -v`

Expected: FAIL because the new card attributes and geometry do not exist.

- [ ] **Step 3: Implement the three page arrangements**

Grid Control with weights 3:2. Put Trigger, Modifier, and Hotkey in
`control_bindings_card`; device details and Preset in `control_device_card`.

Grid Motion with weights 3:2. Put only Strength and Jitter Rate in
`motion_hero_card`; place the existing `motion_summary_var` label in
`motion_summary_card`. Keep remaining numeric controls in Advanced. Use the
existing `_replace_motion_snapshot()` update path without adding traces or a
second snapshot source.

Keep Advanced Canvas, content window, scrollbar, wheel routing, two-column
numeric grid, waveform, and curve behavior unchanged apart from width/padding.

- [ ] **Step 4: Run UI and full suites**

Run: `python -m unittest discover -s tests -p test_ui.py -v`

Expected: PASS.

Run: `python -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: arrange split console workspaces"
```

### Task 4: Polish, Verify, and Smoke Test the Split Console

**Files:**
- Modify: `ui.py`
- Modify: `liquid_widgets.py`
- Modify: `tests/test_ui.py`
- Modify: `tests/test_liquid_widgets.py`
- Modify: `README.md` if active layout wording is stale

**Interfaces:**
- Consumes: completed split console.
- Produces: verified theme, accessibility, visibility, scrolling, and teardown behavior with clean active documentation.

- [ ] **Step 1: Add final cross-page and teardown regression tests**

Assert both themes preserve rail/workspace Canvas tags and contrast; every page
keeps rail actions/footer/runtime/STOP mapped; vertical navigation animation and
all slider/tooltips cancel before service close; changing page and theme does
not reset `motion_summary_var`, settings variables, or outer geometry.

Use existing test fixtures and semantic Canvas tags. Do not create extra Tk
interpreters in worker threads.

- [ ] **Step 2: Run the focused tests and observe any RED gaps**

Run: `python -m unittest discover -s tests -p test_ui.py -k split -v`

Run: `python -m unittest discover -s tests -p test_liquid_widgets.py -k vertical -v`

Expected: any missing integration behavior fails before its production fix;
already-satisfied assertions pass and require no duplicate code.

- [ ] **Step 3: Implement only observed gaps and update active wording**

Adjust spacing, Canvas coordinates, focus order, palette propagation, or
callback cancellation only where focused tests identify a gap. Update README
from top-navigation Control Deck wording to the Liquid Split Console. Do not
edit historical design/plan files.

- [ ] **Step 4: Run required verification**

```powershell
python -m py_compile main.py ui.py motion.py makcu_service.py hotkeys.py settings.py
python -m unittest discover -s tests -v
python -c "import makcu"
git diff --check
```

Expected: every command exits `0`, complete suite reports `OK`, and active
source remains free of retired XP names.

- [ ] **Step 5: Perform safe source UI smoke check**

Run `python main.py`. Inspect Control, Motion, Advanced, dark/light themes,
vertical keyboard navigation, Advanced scrolling, persistent rail actions,
footer, runtime dock, and STOP. Do not invoke Enable, Reconnect, or Test 3s.
Close normally and confirm the process exits promptly.

- [ ] **Step 6: Commit final polish**

```powershell
git add ui.py liquid_widgets.py tests/test_ui.py tests/test_liquid_widgets.py README.md
git commit -m "chore: finalize Liquid Split Console layout"
```

- [ ] **Step 7: Report verification boundary**

Record that intentional physical connection, Trigger/Modifier movement,
Reconnect, Test 3s, global hotkey, STOP, disconnect, and shutdown require a
connected Makcu device. Do not run Nuitka or `gen.bat`.
