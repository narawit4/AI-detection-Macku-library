# Windows XP UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing Cyber Minimal dashboard with a compact `640x560` Windows XP Luna Blue interface while preserving all runtime behavior.

**Architecture:** Keep UI construction and Tk state in `ui.py`, changing only shared presentation constants, ttk styles, widget grouping, and control placement. Preserve every existing motion variable and runtime callback; tests in `tests/test_ui.py` define the new geometry, always-visible controls, Advanced ownership, and unchanged interaction seams.

**Tech Stack:** Python 3, Tkinter/ttk, `unittest`

**Spec:** `docs/superpowers/specs/2026-08-24-windows-xp-ui-redesign.md`

## Global Constraints

- Windows-only fixed-size English desktop application.
- Window geometry is exactly `640x560` and is not resizable.
- Use application-owned Tkinter/ttk styling; do not depend on an OS-provided XP theme.
- Do not add image assets, third-party UI packages, or runtime dependencies.
- Keep Tk widget and Tk-variable access on the main thread.
- Keep STOP visible while Advanced Settings is expanded.
- Every numeric setting retains both a slider and an exact-value input.
- Preserve existing runtime, Makcu, motion, hotkey, persistence, reconnect, Test 3s, and shutdown behavior.
- Preserve unrelated working-tree changes already present in `ui.py`, `motion.py`, `tests/test_ui.py`, and `tests/test_motion.py`.
- Do not run Nuitka or edit generated output.

## File Structure

- Modify `tests/test_ui.py`: layout and widget-ownership acceptance tests for the XP redesign.
- Modify `ui.py`: Luna Blue constants/styles, compact dashboard hierarchy, Advanced control relocation, and scrolling behavior.

---

### Task 1: Lock down the compact dashboard contract

**Files:**
- Modify: `tests/test_ui.py:94-124`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `JitterApp`, public widget attributes created by `ui.py`, and `widget_texts(widget)`.
- Produces: layout expectations for exact geometry, Luna styling, main/Advanced ownership, and STOP visibility.

- [ ] **Step 1: Replace the old minimum-size test with an exact geometry test**

```python
def test_window_is_fixed_size_compact_xp_dashboard(self):
    self.app.update_idletasks()
    self.assertEqual(tuple(map(int, self.app.resizable())), (0, 0))
    self.assertEqual(self.app.geometry().split("+")[0], "640x560")
    self.assertEqual(self.app.cget("background"), "#ECE9D8")
```

- [ ] **Step 2: Add tests for the compact and Advanced control split**

```python
def _is_descendant(widget, ancestor):
    current = widget
    while current is not self.app:
        if current is ancestor:
            return True
        current = current.master
    return False

def test_compact_dashboard_keeps_only_primary_motion_controls(self):
    self.assertFalse(self._is_descendant(self.app.motion_strength_pps_entry,
                                         self.app.advanced_frame))
    self.assertFalse(self._is_descendant(self.app.jitter_rate_hz_entry,
                                         self.app.advanced_frame))
    for key in ("motion_angle_deg", "horizontal_jitter_pps",
                "vertical_jitter_pps"):
        self.assertTrue(self._is_descendant(getattr(self.app, f"{key}_entry"),
                                            self.app.advanced_frame))
    self.assertTrue(self._is_descendant(self.app.hotkey_button,
                                        self.app.advanced_frame))

def test_runtime_group_keeps_stop_always_visible(self):
    self.assertTrue(self._is_descendant(self.app.stop_button,
                                        self.app.runtime_frame))
    self.assertFalse(self._is_descendant(self.app.stop_button,
                                         self.app.advanced_frame))
```

Implement `_is_descendant` as a method on `JitterLayoutTests` so it can access `self.app`.

- [ ] **Step 3: Add a Luna Blue style assertion**

```python
def test_luna_blue_styles_are_registered(self):
    style = ttk.Style(self.app)
    self.assertEqual(style.lookup("XP.Title.TFrame", "background"), "#0054E3")
    self.assertEqual(style.lookup("XP.Group.TLabelframe", "background"), "#ECE9D8")
    self.assertEqual(style.lookup("XP.Danger.TButton", "foreground"), "#A00000")
```

Add `from tkinter import ttk` beside the existing Tk import.

- [ ] **Step 4: Run the focused tests and verify the expected failures**

Run:

```powershell
python -m unittest tests.test_ui.JitterLayoutTests -v
```

Expected: failures for `720x680`, the dark background, missing `XP.*` styles, missing `runtime_frame`, and controls still assigned to their old groups.

- [ ] **Step 5: Commit the failing acceptance tests**

```powershell
git add tests/test_ui.py
git commit -m "test: define compact Windows XP dashboard"
```

Before committing, inspect `git diff -- tests/test_ui.py` and retain all unrelated pre-existing test edits.

---

### Task 2: Build the Windows XP Luna Blue shell

**Files:**
- Modify: `ui.py:25-245`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: the existing `JitterApp` constructor, `_build_page()`, Tk variables, connection labels, and callback methods.
- Produces: `XP_WINDOW = "#ECE9D8"`, `XP_BLUE = "#0054E3"`, registered `XP.*` ttk styles, exact geometry `640x560`, and the public `runtime_frame` widget.

- [ ] **Step 1: Replace the Cyber Minimal presentation constants with Luna Blue constants**

Use a centralized palette and Tahoma typography:

```python
XP_WINDOW = "#ECE9D8"
XP_PANEL = "#FFFFFF"
XP_BLUE = "#0054E3"
XP_BLUE_LIGHT = "#3C8CF0"
XP_BORDER = "#7F9DB9"
XP_TEXT = "#111111"
XP_MUTED = "#555555"
XP_GREEN = "#287025"
XP_AMBER = "#A76500"
XP_RED = "#A00000"

FONT_FAMILY = "Tahoma"
BODY_FONT = (FONT_FAMILY, 8)
SMALL_FONT = (FONT_FAMILY, 8)
TITLE_FONT = (FONT_FAMILY, 9, "bold")
SECTION_FONT = (FONT_FAMILY, 8)
```

- [ ] **Step 2: Change only the window presentation contract**

```python
self.geometry("640x560")
self.configure(background=XP_WINDOW)
```

Retain the fixed-size call, close protocol, initialization order, and runtime setup.

- [ ] **Step 3: Register the Luna Blue ttk styles**

Create and consistently use these styles in `_configure_styles()`:

```python
style.configure("XP.App.TFrame", background=XP_WINDOW)
style.configure("XP.Title.TFrame", background=XP_BLUE)
style.configure("XP.Title.TLabel", background=XP_BLUE, foreground="#FFFFFF",
                font=TITLE_FONT)
style.configure("XP.Group.TLabelframe", background=XP_WINDOW,
                foreground="#164E9E", bordercolor="#ACA899",
                relief="groove", borderwidth=1)
style.configure("XP.Group.TLabelframe.Label", background=XP_WINDOW,
                foreground="#164E9E", font=SECTION_FONT)
style.configure("XP.Group.TLabel", background=XP_WINDOW,
                foreground=XP_TEXT, font=BODY_FONT)
style.configure("XP.Muted.TLabel", background=XP_WINDOW,
                foreground=XP_MUTED, font=SMALL_FONT)
style.configure("XP.Primary.TButton", foreground=XP_TEXT,
                font=(FONT_FAMILY, 8, "bold"), padding=(10, 4))
style.configure("XP.Secondary.TButton", foreground=XP_TEXT,
                font=BODY_FONT, padding=(8, 3))
style.configure("XP.Danger.TButton", foreground=XP_RED,
                font=(FONT_FAMILY, 8, "bold"), padding=(10, 4))
```

Keep dedicated connection styles using `XP_GREEN`, `XP_AMBER`, and `XP_RED`, plus entry, invalid-entry, and readonly-combobox styles with white fields and `XP_BORDER`.

- [ ] **Step 4: Rebuild the header and group framing without changing callbacks**

Use an `XP.Title.TFrame` title strip with `Jitter — Makcu Control`, followed by a connection summary row. Update `_card()` to return `XP.Group.TLabelframe`. In `_build_main_control_card()`, assign the returned group to `self.runtime_frame`, keep `enable_button` and `stop_button` in it, and show the runtime state between them.

Do not alter these callback bindings:

```python
command=self.toggle_enabled
command=self.emergency_stop
command=self.reconnect
```

- [ ] **Step 5: Run focused layout tests**

Run:

```powershell
python -m unittest tests.test_ui.JitterLayoutTests -v
```

Expected: geometry, palette, Luna styles, required actions, runtime ownership, and fixed outer size pass; compact/Advanced ownership may remain failing until Task 3.

- [ ] **Step 6: Commit the XP shell**

```powershell
git add ui.py
git commit -m "feat: add Windows XP Luna dashboard shell"
```

Before committing, inspect `git diff -- ui.py` and retain all unrelated pre-existing UI edits.

---

### Task 3: Compact the primary controls and relocate secondary controls

**Files:**
- Modify: `ui.py:246-407`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `_numeric_control(parent, row, column, label, key, low, high, resolution)`, all existing `motion_vars`, `capture_hotkey()`, `apply_preset()`, `test_run()`, and `toggle_advanced()`.
- Produces: a Quick Jitter group containing only Strength and Jitter Rate; an Advanced group owning Hotkey, Angle, Horizontal, Vertical, and all existing advanced controls.

- [ ] **Step 1: Keep Setup compact**

In `_build_trigger_card()`, keep only Trigger and Modifier. In `_build_action_card()`, place Preset and Test 3s together. Do not construct `hotkey_button` in either group.

- [ ] **Step 2: Reduce Quick Jitter to the approved two controls**

```python
controls = (
    ("Strength", "motion_strength_pps", 0, 500, 1),
    ("Jitter Rate", "jitter_rate_hz", 0.1, 60, 0.1),
)
```

Keep each slider paired with its exact-value entry through `_numeric_control()`.

- [ ] **Step 3: Put secondary controls at the start of Advanced Settings**

Construct `self.hotkey_button` inside `self.advanced_frame`, followed by:

```python
self._numeric_control(grid, 1, 0, "Angle", "motion_angle_deg", 0, 360, 1)
self._numeric_control(grid, 1, 1, "Horizontal", "horizontal_jitter_pps", 0, 500, 1)
self._numeric_control(grid, 2, 0, "Vertical", "vertical_jitter_pps", 0, 500, 1)
```

Then place Randomness, Axis Phase, Smoothness, Ramp, Update Rate, Max Step, Acceleration, Deceleration, Waveform, and Motion Curve in subsequent rows. Retain the exact existing variable keys, ranges, resolution values, callbacks, and public widget attributes.

- [ ] **Step 4: Preserve internal scrolling and STOP access**

Keep the root Canvas and vertical scrollbar. On expansion, pack `advanced_frame` before `advanced_toggle`, update idle geometry, refresh the scroll region, and scroll only the Canvas content; never change root geometry or reparent `runtime_frame`.

- [ ] **Step 5: Run all UI tests**

Run:

```powershell
python -m unittest tests.test_ui -v
```

Expected: all UI layout and runtime tests pass.

- [ ] **Step 6: Commit the compact control hierarchy**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: compact Jitter controls into advanced settings"
```

Commit only changes belonging to this redesign; preserve unrelated working-tree edits.

---

### Task 4: Verify the complete application

**Files:**
- Verify: `main.py`
- Verify: `ui.py`
- Verify: `motion.py`
- Verify: `makcu_service.py`
- Verify: `hotkeys.py`
- Verify: `settings.py`
- Test: `tests/`

**Interfaces:**
- Consumes: the completed `JitterApp` and existing application entry point.
- Produces: verification evidence for syntax, all hardware-free tests, Makcu import, and Windows source UI behavior.

- [ ] **Step 1: Compile every application module**

Run:

```powershell
python -m py_compile main.py ui.py motion.py makcu_service.py hotkeys.py settings.py
```

Expected: exit code 0 with no output.

- [ ] **Step 2: Run the complete automated suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass with no leaked Tk window or worker.

- [ ] **Step 3: Verify the runtime dependency import**

Run:

```powershell
python -c "import makcu"
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Perform a source UI smoke check**

Run `python main.py` on Windows and verify the following before closing the window:

- the window is exactly `640x560` and cannot be resized;
- Luna Blue title treatment and warm XP surface are readable;
- Runtime, Setup, Strength, Jitter Rate, Test 3s, and footer fit compactly;
- Advanced expands within the same outer dimensions and scrolls;
- Hotkey and all secondary motion settings appear in Advanced;
- STOP remains visible and immediately cancels an active test or movement;
- keyboard focus and disabled/active button states remain readable.

If no Makcu device is attached, record hardware interaction as not verified rather than treating Disconnected as a failure.

- [ ] **Step 5: Review the final diff and commit any verification-only corrections**

```powershell
git diff --check
git status --short
git diff -- ui.py tests/test_ui.py
```

If corrections were necessary, run Steps 1-3 again and commit only the redesign files:

```powershell
git add ui.py tests/test_ui.py
git commit -m "fix: polish Windows XP dashboard"
```
