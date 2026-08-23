# Command Center Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Jitter's vertically stacked dashboard with the approved fixed-left, scrollable-right Windows XP Command Center layout while preserving all application behavior.

**Architecture:** Rebuild `JitterApp._build_page()` around four fixed vertical regions and a two-column central body. The left column owns setup and tools; the right column alone owns the Canvas that contains Quick Jitter and the collapsible two-column Advanced Settings card. Existing callbacks, variables, public widget attributes, XP styles, and runtime services remain unchanged.

**Tech Stack:** Python 3, Tkinter/ttk, existing `XPGlossySlider`, `unittest`

**Spec:** `docs/superpowers/specs/2026-08-24-command-center-layout-design.md`

## Global Constraints

- Keep the native title exactly `Jitter — Makcu Control`.
- Keep the outer geometry exactly `640x560` and non-resizable.
- Keep Jitter Disabled and Advanced Settings collapsed on every launch.
- Keep the status strip, left setup/tools column, footer, Runtime State, Enable, and STOP outside the scrollable right workspace.
- STOP must remain fully visible at every right-workspace scroll position.
- Apply right-column scrolling only while the pointer is over the right workspace.
- Keep Quick Jitter above Advanced Settings when Advanced expands.
- Keep Advanced numeric controls in two columns and Waveform/Motion Curve as full-width choice rows.
- Preserve every existing widget callback, Tk variable, public `self.<key>_entry` and `self.<key>_scale` attribute, validation path, preset sync, save debounce, and runtime behavior.
- Keep all Tk access on the main thread and use no new worker.
- Do not change motion, Makcu, hotkey, settings, or configuration schema behavior.
- Do not add assets, dependencies, Pillow, third-party UI packages, or generated output.
- Do not run Nuitka.

## File Structure

- Modify `ui.py`: own the complete Command Center widget hierarchy, right-only scrolling, Advanced expansion, and final layout polish.
- Modify `tests/test_ui.py`: replace obsolete stacked-layout assertions and add hardware-free hierarchy, scroll-scope, grid, synchronization, and viewport tests.
- No new runtime module is required; this is a presentation-only reorganization of the existing `JitterApp` page.

---

### Task 1: Build the Command Center shell and move controls to their approved regions

**Files:**
- Modify: `tests/test_ui.py:78-310`
- Modify: `ui.py:245-439`

**Interfaces:**
- Consumes: existing `JitterApp` variables, callbacks, ttk styles, `_numeric_control()`, and `XPGlossySlider`.
- Produces: widget attributes `shell`, `status_strip`, `command_center`, `left_column`, `right_host`, `right_canvas`, `right_scrollbar`, `right_content`, `footer_frame`, `runtime_frame`, `setup_frame`, and `tools_frame`; compatibility aliases `canvas` and `content` continue to point to `right_canvas` and `right_content`.

- [ ] **Step 1: Replace obsolete stacked-layout tests with failing Command Center hierarchy tests**

In `JitterLayoutTests`, retain `_is_descendant()` and replace
`test_compact_dashboard_keeps_only_primary_motion_controls()`,
`test_setup_group_combines_bindings_preset_and_test_run()`, and
`test_runtime_group_uses_the_approved_title()` with these behavior tests:

```python
def test_command_center_uses_fixed_status_body_footer_runtime_order(self):
    regions = (
        self.app.status_strip,
        self.app.command_center,
        self.app.footer_frame,
        self.app.runtime_frame,
    )
    self.assertTrue(all(widget.master is self.app.shell for widget in regions))
    self.assertEqual(
        [int(widget.grid_info()["row"]) for widget in regions],
        [0, 1, 2, 3],
    )
    self.assertEqual(self.app.shell.grid_rowconfigure(1)["weight"], 1)

def test_setup_and_tools_stay_in_fixed_left_column(self):
    fixed_widgets = (
        self.app.trigger_combo,
        self.app.modifier_combo,
        self.app.preset_combo,
        self.app.hotkey_button,
        self.app.reconnect_button,
        self.app.test_button,
        self.app.advanced_toggle,
    )
    for widget in fixed_widgets:
        with self.subTest(widget=str(widget)):
            self.assertTrue(self._is_descendant(widget, self.app.left_column))
            self.assertFalse(self._is_descendant(widget, self.app.right_content))

def test_quick_and_advanced_controls_live_in_right_workspace(self):
    right_widgets = (
        self.app.motion_strength_pps_entry,
        self.app.jitter_rate_hz_entry,
        self.app.motion_angle_deg_entry,
        self.app.waveform_combo,
        self.app.motion_curve_combo,
    )
    for widget in right_widgets:
        with self.subTest(widget=str(widget)):
            self.assertTrue(self._is_descendant(widget, self.app.right_content))
    self.assertFalse(self._is_descendant(self.app.hotkey_button,
                                         self.app.advanced_frame))

def test_status_strip_combines_device_and_connection_state(self):
    self.assertTrue(self._is_descendant(self.app.device_label,
                                        self.app.status_strip))
    self.assertTrue(self._is_descendant(self.app.connection_label,
                                        self.app.status_strip))
    self.assertFalse(self._is_descendant(self.app.reconnect_button,
                                         self.app.status_strip))
```

Keep existing tests for the title, palette, buttons, required actions, glossy
sliders, geometry, and runtime behavior.

- [ ] **Step 2: Run the focused hierarchy tests and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_ui.py -k command_center -k fixed_left -k right_workspace -k status_strip -v
```

Expected: errors for missing `status_strip`, `command_center`, `left_column`,
`right_content`, and related attributes.

- [ ] **Step 3: Replace `_build_page()` with the four-region shell**

Implement the shell with fixed rows and a weighted body row:

```python
def _build_page(self) -> None:
    self.shell = ttk.Frame(self, style="XP.App.TFrame", padding=(8, 8, 8, 8))
    self.shell.pack(fill="both", expand=True)
    self.shell.columnconfigure(0, weight=1)
    self.shell.rowconfigure(1, weight=1)

    self.status_strip = ttk.Frame(self.shell, style="XP.Status.TFrame",
                                  padding=(7, 5))
    self.status_strip.grid(row=0, column=0, sticky="ew", pady=(0, 7))
    self._build_status_strip()

    self.command_center = ttk.Frame(self.shell, style="XP.App.TFrame")
    self.command_center.grid(row=1, column=0, sticky="nsew")
    self.command_center.rowconfigure(0, weight=1)
    self.command_center.columnconfigure(0, minsize=190)
    self.command_center.columnconfigure(1, weight=1)

    self.left_column = ttk.Frame(self.command_center, style="XP.App.TFrame")
    self.left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
    self._build_trigger_card()
    self._build_action_card()

    self._build_right_workspace()
    self._build_quick_card()
    self._build_advanced_card()
    self._build_footer()
    self._build_main_control_card()
```

Delete the old full-window `fixed_content` plus full-window `scroll_host`
construction. Do not retain two competing Canvases.

- [ ] **Step 4: Add the compact status strip and right workspace constructors**

Replace `_build_header()` and `_build_device_card()` with:

```python
def _build_status_strip(self) -> None:
    self.device_label = ttk.Label(
        self.status_strip,
        textvariable=self.device_status_var,
        style="XP.Muted.TLabel",
    )
    self.device_label.pack(side="left", fill="x", expand=True)
    ttk.Label(self.status_strip, text="Connection:",
              style="XP.Muted.TLabel").pack(side="left", padx=(8, 4))
    self.connection_label = ttk.Label(
        self.status_strip,
        textvariable=self.connection_status_var,
        style="XP.StatusDisconnected.TLabel",
    )
    self.connection_label.pack(side="right")

def _build_right_workspace(self) -> None:
    self.right_host = ttk.Frame(self.command_center, style="XP.App.TFrame")
    self.right_host.grid(row=0, column=1, sticky="nsew")
    self.right_host.rowconfigure(0, weight=1)
    self.right_host.columnconfigure(0, weight=1)
    self.right_canvas = tk.Canvas(
        self.right_host,
        background=XP_WINDOW,
        highlightthickness=0,
        borderwidth=0,
    )
    self.right_scrollbar = ttk.Scrollbar(
        self.right_host,
        orient="vertical",
        command=self.right_canvas.yview,
    )
    self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)
    self.right_canvas.grid(row=0, column=0, sticky="nsew")
    self.right_scrollbar.grid(row=0, column=1, sticky="ns")
    self.right_content = ttk.Frame(
        self.right_canvas,
        style="XP.App.TFrame",
        padding=(0, 0, 4, 8),
    )
    self.right_content_window = self.right_canvas.create_window(
        (0, 0),
        window=self.right_content,
        anchor="nw",
    )
    self.canvas = self.right_canvas
    self.content = self.right_content
    self.content_window = self.right_content_window
    self.right_content.bind("<Configure>", self._refresh_scrollregion)
    self.right_canvas.bind("<Configure>", self._resize_content_window)
```

Update `_refresh_scrollregion()` and `_resize_content_window()` to operate on
`right_canvas` and `right_content_window`. The compatibility aliases exist for
older callers, not as the primary implementation path.

- [ ] **Step 5: Move setup, tools, footer, and runtime widgets to fixed regions**

Keep `_card(title, parent)` but require each builder to pass its parent.
Refactor `_build_trigger_card()` so `setup_frame` is a `Control Setup` card in
`left_column`. Stack Trigger, Modifier, Preset, and Hotkey as label/control rows:

```python
self.setup_frame = self._card("Control Setup", self.left_column)
self.setup_frame.columnconfigure(0, weight=1)

def combo_row(row, label, variable, values, width):
    ttk.Label(self.setup_frame, text=label,
              style="XP.Group.TLabel").grid(row=row, column=0, sticky="w")
    combo = ttk.Combobox(
        self.setup_frame,
        textvariable=variable,
        values=values,
        state="readonly",
        style="XP.TCombobox",
        width=width,
    )
    combo.grid(row=row + 1, column=0, sticky="ew", pady=(2, 6))
    return combo
```

Use it to assign `trigger_combo`, `modifier_combo`, and `preset_combo`, then
bind their existing callbacks. Use these exact assignments and retain the
existing choice values:

```python
self.trigger_combo = combo_row(
    0, "Trigger", self.trigger_var,
    ("Left", "Right", "Middle", "Mouse4", "Mouse5"), 14,
)
self.trigger_combo.bind("<<ComboboxSelected>>", self._bindings_event)
self.modifier_combo = combo_row(
    2, "Modifier", self.modifier_var,
    ("None", "Left", "Right", "Middle", "Mouse4", "Mouse5"), 14,
)
self.modifier_combo.bind("<<ComboboxSelected>>", self._bindings_event)
self.preset_combo = combo_row(
    4, "Preset", self.preset_var, self.preset_values, 14,
)
self.preset_combo.bind("<<ComboboxSelected>>", self.apply_preset)
self.hotkey_button = ttk.Button(
    self.setup_frame,
    text=f"Hotkey: {self.hotkey_name_var.get()}",
    style="XP.Secondary.TButton",
    command=self.capture_hotkey,
)
self.hotkey_button.grid(row=6, column=0, sticky="ew", pady=(2, 0))
```

Refactor `_build_action_card()` to create `tools_frame = _card("Tools",
self.left_column)` and place these existing controls vertically with `fill="x"`:

```python
self.reconnect_button = ttk.Button(
    self.tools_frame,
    text="Reconnect",
    style="XP.Secondary.TButton",
    command=self.reconnect,
)
self.test_button = ttk.Button(
    self.tools_frame,
    text="Test 3s",
    style="XP.Secondary.TButton",
    command=self.test_run,
)
self.advanced_toggle = ttk.Button(
    self.tools_frame,
    text="Advanced Settings ▼",
    style="XP.Secondary.TButton",
    command=self.toggle_advanced,
)
for button in (self.reconnect_button, self.test_button, self.advanced_toggle):
    button.pack(fill="x", pady=(0, 5))
```

Change `_build_footer()` to create `footer_frame` directly under `shell` at row
2. Change `_build_main_control_card()` to create `runtime_frame` directly under
`shell` at row 3, configure three weighted columns, and grid Enable, the existing
center state frame, and STOP into columns 0, 1, and 2. Preserve every callback,
text variable, and button style.

Use this fixed footer construction:

```python
def _build_footer(self) -> None:
    self.footer_frame = ttk.Frame(self.shell, style="XP.App.TFrame")
    self.footer_frame.grid(row=2, column=0, sticky="ew", pady=(6, 3))
    self.footer_label = ttk.Label(
        self.footer_frame,
        textvariable=self.footer_var,
        style="XP.Muted.TLabel",
        anchor="w",
    )
    self.footer_label.pack(fill="x")
```

In the runtime builder, construct the existing buttons directly under
`runtime_frame`, preserving `toggle_enabled` and `emergency_stop` as commands.
Create the center `state` frame directly under `runtime_frame` and retain the
`RUNTIME` label plus `runtime_status_var` label.

- [ ] **Step 6: Point Quick and Advanced cards at `right_content`**

Call `_card("Quick Jitter", self.right_content)` and retain the existing Quick
numeric controls. Construct `advanced_frame` with `right_content` as parent,
remove the old Hotkey construction from `_build_advanced_card()`, and do not pack
`advanced_frame` during initial construction. The Advanced toggle now already
exists in `tools_frame`.

- [ ] **Step 7: Run focused and complete UI tests**

Run:

```powershell
python -m unittest discover -s tests -p test_ui.py -k command_center -k fixed_left -k right_workspace -k status_strip -v
python -m unittest discover -s tests -p test_ui.py -v
```

Expected: all hierarchy and existing UI/runtime tests pass. Update only obsolete
assertions that explicitly depended on the removed stacked hierarchy; do not
weaken runtime behavior assertions.

- [ ] **Step 8: Commit the Command Center shell**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: rebuild UI as XP Command Center"
```

---

### Task 2: Implement Advanced expansion and right-workspace-only scrolling

**Files:**
- Modify: `tests/test_ui.py`
- Modify: `ui.py:430-490`

**Interfaces:**
- Consumes: Task 1 `right_host`, `right_canvas`, `right_content`, `advanced_frame`, `advanced_toggle`, and `_advanced_visible`.
- Produces: `toggle_advanced()`, `_is_descendant_of(widget, ancestor) -> bool`, and `_on_right_mousewheel(event) -> str | None` with right-only scrolling and collapse-to-top behavior.

- [ ] **Step 1: Add failing Advanced default and placement tests**

Add to `JitterLayoutTests`:

```python
def test_advanced_starts_collapsed_and_expands_below_quick(self):
    self.assertFalse(self.app._advanced_visible)
    self.assertFalse(self.app.advanced_state_var.get())
    self.assertFalse(self.app.advanced_frame.winfo_manager())
    self.assertEqual(self.app.advanced_toggle.cget("text"),
                     "Advanced Settings ▼")

    self.app.toggle_advanced()
    self.app.update_idletasks()

    self.assertTrue(self.app._advanced_visible)
    self.assertEqual(self.app.advanced_frame.winfo_manager(), "pack")
    children = self.app.right_content.pack_slaves()
    self.assertLess(children.index(self.app.quick_frame),
                    children.index(self.app.advanced_frame))
    self.assertEqual(self.app.advanced_toggle.cget("text"),
                     "Advanced Settings ▲")

def test_collapsing_advanced_returns_right_workspace_to_top(self):
    self.app.deiconify()
    self.app.toggle_advanced()
    self.app.update()
    self.app.right_canvas.yview_moveto(1.0)
    self.app.toggle_advanced()
    self.app.update()
    self.assertEqual(self.app.right_canvas.yview()[0], 0.0)
```

Assign `self.quick_frame` in `_build_quick_card()` so the first test asserts
observable layout order without searching labels.

- [ ] **Step 2: Add a failing right-only mouse-wheel routing test**

Add `from types import SimpleNamespace` to `tests/test_ui.py`, then add:

```python
def test_mousewheel_scrolls_only_over_right_workspace(self):
    self.app.deiconify()
    self.app.toggle_advanced()
    self.app.update()
    self.app.right_canvas.yview_moveto(0.0)

    right_event = SimpleNamespace(
        delta=-120,
        x_root=self.app.right_canvas.winfo_rootx() + 10,
        y_root=self.app.right_canvas.winfo_rooty() + 10,
    )
    self.assertEqual(self.app._on_right_mousewheel(right_event), "break")
    self.app.update_idletasks()
    after_right = self.app.right_canvas.yview()[0]
    self.assertGreater(after_right, 0.0)

    left_event = SimpleNamespace(
        delta=-120,
        x_root=self.app.left_column.winfo_rootx() + 10,
        y_root=self.app.left_column.winfo_rooty() + 10,
    )
    self.assertIsNone(self.app._on_right_mousewheel(left_event))
    self.assertEqual(self.app.right_canvas.yview()[0], after_right)
```

- [ ] **Step 3: Run the new Advanced/scroll tests and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_ui.py -k advanced_starts -k collapsing_advanced -k mousewheel -v
```

Expected: failures because `quick_frame`, arrow text, collapse reset, and
`_on_right_mousewheel()` do not yet implement the approved behavior.

- [ ] **Step 4: Implement deterministic Advanced expansion**

Store the Quick card and replace `toggle_advanced()`:

```python
def _build_quick_card(self) -> None:
    self.quick_frame = self._card("Quick Jitter", self.right_content)
    grid = ttk.Frame(self.quick_frame, style="XP.App.TFrame")
    # retain the existing Strength and Jitter Rate construction

def toggle_advanced(self) -> None:
    if self._advanced_visible:
        self.advanced_frame.pack_forget()
        self._advanced_visible = False
        self.advanced_state_var.set(False)
        self.advanced_toggle.configure(text="Advanced Settings ▼")
        self.right_canvas.yview_moveto(0.0)
    else:
        self.advanced_frame.pack(fill="x", pady=(0, 9))
        self._advanced_visible = True
        self.advanced_state_var.set(True)
        self.advanced_toggle.configure(text="Advanced Settings ▲")
    self.update_idletasks()
    self._refresh_scrollregion()
```

Because `quick_frame` is packed before the initially hidden `advanced_frame`,
packing Advanced places it immediately below Quick. No `before=` reference to
the fixed left-column toggle is valid or needed.

- [ ] **Step 5: Implement root-bound right-only mouse-wheel routing**

Bind once after page construction:

```python
self.bind("<MouseWheel>", self._on_right_mousewheel, add="+")
```

Add these methods:

```python
@staticmethod
def _is_descendant_of(widget: tk.Misc | None, ancestor: tk.Misc) -> bool:
    current = widget
    while current is not None:
        if current is ancestor:
            return True
        current = getattr(current, "master", None)
    return False

def _on_right_mousewheel(self, event) -> str | None:
    target = self.winfo_containing(event.x_root, event.y_root)
    if not self._is_descendant_of(target, self.right_host):
        return None
    bounds = self.right_canvas.bbox("all")
    if bounds is None or bounds[3] <= self.right_canvas.winfo_height():
        return None
    direction = -1 if event.delta > 0 else 1
    self.right_canvas.yview_scroll(direction, "units")
    return "break"
```

This avoids `bind_all()` ownership and leaves wheel events elsewhere in the
window untouched.

- [ ] **Step 6: Run focused tests and complete UI tests**

Run:

```powershell
python -m unittest discover -s tests -p test_ui.py -k advanced_starts -k collapsing_advanced -k mousewheel -v
python -m unittest discover -s tests -p test_ui.py -v
```

Expected: all tests pass without Tk callback warnings.

- [ ] **Step 7: Commit Advanced and scrolling behavior**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: add right-only Advanced workspace scrolling"
```

---

### Task 3: Apply the approved G2 grid and fixed runtime polish

**Files:**
- Modify: `tests/test_ui.py`
- Modify: `ui.py:130-210,354-439`

**Interfaces:**
- Consumes: Task 1 fixed regions and Task 2 expandable right content.
- Produces: `quick_grid`, `advanced_grid`, dense two-column Advanced placement, full-width choice rows, and equal-weight fixed Enable/STOP controls.

- [ ] **Step 1: Add a failing literal Advanced grid placement test**

Add:

```python
def test_advanced_uses_approved_two_column_grid(self):
    expected_positions = {
        "motion_angle_deg": (0, 0),
        "horizontal_jitter_pps": (0, 1),
        "vertical_jitter_pps": (1, 0),
        "jitter_randomness_percent": (1, 1),
        "jitter_axis_phase_deg": (2, 0),
        "smoothness_percent": (2, 1),
        "ramp_up_ms": (3, 0),
        "update_rate_hz": (3, 1),
        "max_step_px": (4, 0),
        "acceleration_pps2": (4, 1),
        "deceleration_pps2": (5, 0),
    }
    for key, expected in expected_positions.items():
        with self.subTest(key=key):
            block = getattr(self.app, f"{key}_entry").master.master
            info = block.grid_info()
            self.assertEqual((int(info["row"]), int(info["column"])), expected)
            self.assertIs(block.master, self.app.advanced_grid)

def test_advanced_choices_span_the_full_grid_width(self):
    for combo in (self.app.waveform_combo, self.app.motion_curve_combo):
        with self.subTest(combo=str(combo)):
            choice_row = combo.master
            info = choice_row.grid_info()
            self.assertEqual(int(info["columnspan"]), 2)
            self.assertIs(choice_row.master, self.app.advanced_grid)
```

The literal table catches accidental stacking, swapped columns, and regression
to the old sparse grid.

- [ ] **Step 2: Add failing fixed runtime alignment tests**

Add:

```python
def test_runtime_actions_have_equal_fixed_weight(self):
    self.assertIs(self.app.enable_button.master, self.app.runtime_frame)
    self.assertIs(self.app.stop_button.master, self.app.runtime_frame)
    self.assertEqual(int(self.app.enable_button.grid_info()["column"]), 0)
    self.assertEqual(int(self.app.stop_button.grid_info()["column"]), 2)
    self.assertIn("ew", self.app.enable_button.grid_info()["sticky"])
    self.assertIn("ew", self.app.stop_button.grid_info()["sticky"])
    self.assertEqual(self.app.runtime_frame.grid_columnconfigure(0)["weight"],
                     self.app.runtime_frame.grid_columnconfigure(2)["weight"])

def test_footer_and_runtime_are_outside_scrollable_workspace(self):
    for widget in (self.app.footer_frame, self.app.runtime_frame,
                   self.app.enable_button, self.app.stop_button):
        with self.subTest(widget=str(widget)):
            self.assertFalse(self._is_descendant(widget, self.app.right_host))
```

- [ ] **Step 3: Run the G2/runtime tests and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_ui.py -k approved_two_column -k full_grid_width -k equal_fixed_weight -k outside_scrollable -v
```

Expected: grid placement and choice-row tests fail against the sparse existing
Advanced grid; runtime tests fail until the fixed bar uses the approved grid.

- [ ] **Step 4: Rebuild the Advanced grid with dense G2 placement**

Assign `self.advanced_grid` and use this exact control tuple:

```python
self.advanced_grid = ttk.Frame(self.advanced_frame, style="XP.App.TFrame")
self.advanced_grid.pack(fill="x")
self.advanced_grid.columnconfigure(0, weight=1, uniform="advanced")
self.advanced_grid.columnconfigure(1, weight=1, uniform="advanced")
controls = (
    (0, 0, "Angle", "motion_angle_deg", 0, 360, 1),
    (0, 1, "Horizontal", "horizontal_jitter_pps", 0, 500, 1),
    (1, 0, "Vertical", "vertical_jitter_pps", 0, 500, 1),
    (1, 1, "Randomness", "jitter_randomness_percent", 0, 100, 1),
    (2, 0, "Axis Phase", "jitter_axis_phase_deg", 0, 360, 1),
    (2, 1, "Smoothness", "smoothness_percent", 1, 100, 1),
    (3, 0, "Ramp (ms)", "ramp_up_ms", 0, 2000, 1),
    (3, 1, "Update Rate", "update_rate_hz", 20, 500, 1),
    (4, 0, "Max Step", "max_step_px", 1, 50, 1),
    (4, 1, "Acceleration", "acceleration_pps2", 1, 10000, 1),
    (5, 0, "Deceleration", "deceleration_pps2", 1, 10000, 1),
)
for row, column, label, key, low, high, resolution in controls:
    self._numeric_control(
        self.advanced_grid, row, column, label, key, low, high, resolution
    )
```

Create each full-width choice as a frame inside `advanced_grid`:

```python
waveform_row = ttk.Frame(self.advanced_grid, style="XP.App.TFrame")
waveform_row.grid(row=6, column=0, columnspan=2, sticky="ew", padx=5, pady=(8, 3))
ttk.Label(waveform_row, text="Waveform",
          style="XP.Group.TLabel").pack(side="left")
self.waveform_combo = ttk.Combobox(
    waveform_row,
    textvariable=self.motion_vars["jitter_waveform"],
    values=JITTER_WAVEFORMS,
    state="readonly",
    style="XP.TCombobox",
)
self.waveform_combo.pack(side="right", fill="x", expand=True, padx=(8, 0))
```

Repeat with a distinct `curve_row` at row 7 for Motion Curve and
`self.motion_curve_combo`. Do not reuse one frame for both variables.

- [ ] **Step 5: Make Quick and runtime columns visually balanced**

Assign `self.quick_grid` and configure both Quick columns with equal uniform
weight. In `_build_main_control_card()`, grid Enable and STOP with `sticky="ew"`,
set runtime columns 0 and 2 to equal uniform weight, and let center column 1
expand:

```python
self.runtime_frame.columnconfigure(0, weight=1, uniform="runtime_actions")
self.runtime_frame.columnconfigure(1, weight=2)
self.runtime_frame.columnconfigure(2, weight=1, uniform="runtime_actions")
self.enable_button.grid(row=0, column=0, sticky="ew")
state.grid(row=0, column=1, sticky="ew", padx=10)
self.stop_button.grid(row=0, column=2, sticky="ew")
```

Register `XP.Status.TFrame` with a white card surface and cool border-compatible
background. Reuse the existing palette constants; do not add a second palette.

- [ ] **Step 6: Run focused tests and the complete suite**

Run:

```powershell
python -m unittest discover -s tests -p test_ui.py -k approved_two_column -k full_grid_width -k equal_fixed_weight -k outside_scrollable -v
python -m unittest discover -s tests -v
```

Expected: all hardware-free tests pass without Tk warnings or leaked windows.

- [ ] **Step 7: Commit G2 and runtime polish**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: polish Command Center control grid"
```

---

### Task 4: Verify the complete Command Center redesign

**Files:**
- Verify: `ui.py`
- Verify: `xp_widgets.py`
- Test: `tests/test_ui.py`
- Test: `tests/test_xp_widgets.py`
- Test: `tests/`

**Interfaces:**
- Consumes: completed Command Center hierarchy and existing runtime services.
- Produces: fresh syntax, automated, import, and Windows UI smoke evidence.

- [ ] **Step 1: Compile all application modules**

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

Expected: every test passes with no Tk callback warning or leaked window.

- [ ] **Step 3: Verify the runtime dependency import**

Run:

```powershell
python -c "import makcu"
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Perform a Windows source UI smoke check without movement**

Run `python main.py`, do not click Enable or Test 3s, and verify:

- native title is `Jitter — Makcu Control` and geometry remains `640x560`;
- device/connection strip spans the top without an internal brand banner;
- Trigger, Modifier, Preset, Hotkey, Reconnect, Test 3s, and Advanced toggle stay
  fixed in the left column;
- Quick Jitter occupies the top of the right workspace;
- Advanced starts collapsed and expands beneath Quick in a dense two-column
  grid;
- only the right workspace scrolls, and the left column does not move;
- slider hover, pressed, focus, drag, keyboard, and value bubble effects remain
  readable in the narrower G2 cells;
- valid exact-entry changes and presets move the matching sliders silently;
- footer, Enable, Runtime State, and STOP stay fully visible at the top and
  bottom of the right scroll range;
- closing the window exits cleanly.

- [ ] **Step 5: Review final state**

Run:

```powershell
git diff --check
git status --short
git diff -- ui.py tests/test_ui.py
```

Confirm only intended source/test changes remain. If a smoke correction is
needed, add a failing test first, implement the minimal correction, rerun Steps
1-3, and commit:

```powershell
git add ui.py tests/test_ui.py
git commit -m "fix: complete Command Center layout polish"
```
