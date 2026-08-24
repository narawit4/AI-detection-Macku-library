# Liquid XP Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two-column dashboard with a fixed-size, three-page Liquid XP navigation layout while preserving Makcu, motion, configuration, STOP, and shutdown behavior.

**Architecture:** Add a pure-Tk `LiquidXPNav` canvas for tab rendering, keyboard selection, palette changes, and a cancellable 160 ms active-pill animation. `JitterApp` owns three persistent page frames, maps selection to exactly one visible page, and keeps status, mini actions, footer, and runtime controls outside page scrolling.

**Tech Stack:** Python 3.12, Tkinter/ttk, `unittest`, standard-library threading and queues.

**Spec:** `docs/superpowers/specs/2026-08-24-liquid-xp-navigation-design.md`

## Global Constraints

- Windows-only application with a fixed 640x560 outer window.
- Keep all Tk widget and Tk-variable access on the main thread.
- Preserve the thread-safe UI queue; no worker may call Tk.
- STOP, disconnect, hotkey disable, and shutdown must signal cancellation immediately.
- Add no blur, image, Pillow, or non-standard UI package.
- Do not change motion algorithms, Makcu protocol, config schema, or window size.
- Preserve current uncommitted theme, tooltip, shutdown-queue, and mini-icon work.
- Do not run Nuitka unless explicitly requested.

## File Map

- `xp_widgets.py`: add `LiquidXPNav`; retain `XPGlossySlider` behavior.
- `tests/test_xp_widgets.py`: nav selection, keyboard, animation, palette, and teardown tests.
- `ui.py`: navigation shell, three pages, mini actions, theme propagation, scrolling, and shutdown.
- `tests/test_ui.py`: layout and integration tests; remove assumptions tied to the retired two-column layout.
- No planned changes to motion, Makcu, hotkey, settings, or entry-point modules.

---

### Task 1: Navigation Selection and Keyboard Contract

**Files:**
- Modify: `xp_widgets.py`
- Test: `tests/test_xp_widgets.py`

**Interfaces:**
- Produces: `LiquidXPNav(parent, *, labels: tuple[str, ...], command: Callable[[int], None] | None = None, selected: int = 0, palette: Mapping[str, str] | None = None, animation_ms: int = 160, width: int = 330, height: int = 38)`.
- Produces: `selected_index: int`, `select(index: int, *, animate: bool = True, notify: bool = True) -> None`, `set_palette(palette: Mapping[str, str]) -> None`, and `cancel_animation() -> None`.

- [ ] **Step 1: Write failing construction and selection tests**

```python
from types import SimpleNamespace
from xp_widgets import LiquidXPNav, XPGlossySlider

class LiquidXPNavTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.selected = []
        self.nav = LiquidXPNav(
            self.root,
            labels=("Setup", "Motion", "Advanced"),
            command=self.selected.append,
            width=330,
        )
        self.nav.pack()
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def test_select_clamps_index_and_notifies_once(self):
        self.nav.select(2, animate=False)
        self.assertEqual(self.nav.selected_index, 2)
        self.assertEqual(self.selected, [2])
        self.nav.select(99, animate=False)
        self.assertEqual(self.nav.selected_index, 2)
        self.assertEqual(self.selected, [2])

    def test_arrow_keys_select_adjacent_tabs(self):
        self.nav._on_key(1)
        self.nav._on_key(1)
        self.nav._on_key(1)
        self.assertEqual(self.nav.selected_index, 2)
        self.nav._on_key(-1)
        self.assertEqual(self.nav.selected_index, 1)

    def test_pointer_selects_the_hit_tab(self):
        left, right = self.nav._tab_bounds(1)
        self.nav._on_click(SimpleNamespace(x=(left + right) / 2))
        self.assertEqual(self.nav.selected_index, 1)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m unittest discover -s tests -p test_xp_widgets.py -v`

Expected: import failure because `LiquidXPNav` does not exist.

- [ ] **Step 3: Implement the minimal selection widget**

Add a `tk.Canvas` subclass with non-empty labels, clamped selection,
`takefocus=True`, pointer hit testing, and Left/Right bindings. Tag each tab as
`tab-<index>` and draw its label.

```python
def select(self, index: int, *, animate: bool = True,
           notify: bool = True) -> None:
    target = min(len(self.labels) - 1, max(0, int(index)))
    if target == self.selected_index:
        return
    self.selected_index = target
    self._redraw()
    if notify and self.command is not None:
        self.command(target)

def _on_key(self, delta: int) -> str:
    self.select(self.selected_index + delta)
    return "break"
```

- [ ] **Step 4: Run `python -m unittest discover -s tests -p test_xp_widgets.py -v` and verify GREEN**

Expected: all slider and basic nav tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- xp_widgets.py tests/test_xp_widgets.py
git commit -m "feat: add Liquid XP navigation contract"
```

---

### Task 2: Liquid Rendering, Palette, and Animation

**Files:**
- Modify: `xp_widgets.py`
- Test: `tests/test_xp_widgets.py`

**Interfaces:**
- Consumes: Task 1's `LiquidXPNav` public API.
- Produces: `_animation_after_id: str | None`, `_pill_x: float`, `_target_pill_x(index: int) -> float`, and Canvas tags `capsule`, `pill`, `tab`, and `focus`.

- [ ] **Step 1: Write failing palette and animation tests**

```python
def test_palette_redraws_capsule_and_active_pill(self):
    palette = {
        "background": "#171B22", "capsule": "#285A91",
        "capsule_outline": "#8CBCEB", "pill": "#4C8CCC",
        "pill_highlight": "#FFFFFF", "text": "#E7ECF3",
        "active_text": "#FFFFFF", "focus": "#F2B84B",
    }
    self.nav.set_palette(palette)
    self.root.update_idletasks()
    self.assertEqual(self.nav.cget("background"), "#171B22")
    self.assertEqual(self.nav.itemcget("capsule", "fill"), "#285A91")

def test_new_selection_replaces_obsolete_animation(self):
    self.nav.select(2)
    first = self.nav._animation_after_id
    self.nav.select(1)
    self.assertIsNotNone(self.nav._animation_after_id)
    self.assertNotEqual(self.nav._animation_after_id, first)
    self.nav.cancel_animation()
    self.assertIsNone(self.nav._animation_after_id)

def test_destroy_cancels_animation(self):
    self.nav.select(2)
    self.nav.destroy()
    self.assertIsNone(self.nav._animation_after_id)
```

- [ ] **Step 2: Run `python -m unittest discover -s tests -p test_xp_widgets.py -k LiquidXPNav -v` and verify RED**

Expected: failures for missing tagged rendering and animation state.

- [ ] **Step 3: Implement XP capsule rendering and 160 ms interpolation**

Use Canvas rectangles/ovals for rounded ends, outline, top highlight, active
pill, centered text, and focus outline. Animate with ten 16 ms steps. A new
selection cancels the obsolete callback and starts from the current `_pill_x`.

```python
def cancel_animation(self) -> None:
    callback_id = self._animation_after_id
    self._animation_after_id = None
    if callback_id is not None:
        try:
            self.after_cancel(callback_id)
        except tk.TclError:
            pass
```

- [ ] **Step 4: Run `python -m unittest tests/test_xp_widgets.py -v` and verify GREEN**

Expected: all navigation and slider tests pass without Tcl errors.

- [ ] **Step 5: Commit**

```powershell
git add -- xp_widgets.py tests/test_xp_widgets.py
git commit -m "feat: animate Liquid XP navigation"
```

---

### Task 3: Three Persistent Content Pages

**Files:**
- Modify: `ui.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `LiquidXPNav(..., command=self.select_page)`.
- Produces: `nav`, `navigation_frame`, `page_host`, `setup_page`, `motion_page`, `advanced_page`, `pages: tuple[ttk.Frame, ...]`, and `select_page(index: int) -> None`.

- [ ] **Step 1: Replace obsolete layout assertions with failing page tests**

```python
def test_shell_uses_status_navigation_page_footer_runtime_order(self):
    regions = (self.app.status_strip, self.app.navigation_frame,
               self.app.page_host, self.app.footer_frame,
               self.app.runtime_frame)
    self.assertEqual([int(w.grid_info()["row"]) for w in regions],
                     [0, 1, 2, 3, 4])

def test_navigation_owns_three_persistent_pages(self):
    self.assertEqual(self.app.nav.labels, ("Setup", "Motion", "Advanced"))
    self.assertEqual(self.app.pages,
                     (self.app.setup_page, self.app.motion_page,
                      self.app.advanced_page))
    self.assertTrue(self._is_descendant(self.app.trigger_combo,
                                        self.app.setup_page))
    self.assertTrue(self._is_descendant(
        self.app.motion_strength_pps_entry, self.app.motion_page))
    self.assertTrue(self._is_descendant(
        self.app.waveform_combo, self.app.advanced_page))

def test_select_page_shows_one_page_without_resetting_values(self):
    self.app.motion_strength_pps_var.set("123")
    self.app.select_page(2)
    self.assertEqual(self.app.nav.selected_index, 2)
    self.assertEqual(self.app.page_host.grid_slaves(), [self.app.pages[2]])
    self.app.select_page(1)
    self.assertEqual(self.app.motion_strength_pps_var.get(), "123")
```

- [ ] **Step 2: Run `python -m unittest discover -s tests -p test_ui.py -v` and verify RED**

Expected: failures because the navigation frame and pages do not exist.

- [ ] **Step 3: Build the page host and move existing controls**

Import `LiquidXPNav`. Replace `command_center`, fixed left column, and the old
right workspace with `navigation_frame` and `page_host`. Reparent construction
so trigger/preset/hotkey controls target `setup_page`, quick controls target
`motion_page`, and detailed controls target the scrollable `advanced_page`.
Keep every existing Tk variable, widget attribute, command, and binding.

```python
def select_page(self, index: int) -> None:
    selected = min(len(self.pages) - 1, max(0, int(index)))
    for page in self.pages:
        page.grid_remove()
    self.pages[selected].grid()
    if self.nav.selected_index != selected:
        self.nav.select(selected, notify=False)
```

- [ ] **Step 4: Run the UI suite and verify GREEN**

Run: `python -m unittest discover -s tests -p test_ui.py -v`

Expected: new page tests and all unchanged runtime tests pass. Update only tests
that explicitly describe the retired two-column or Tools-card structure.

- [ ] **Step 5: Commit**

```powershell
git add -- ui.py tests/test_ui.py
git commit -m "feat: split dashboard into navigation pages"
```

---

### Task 4: Persistent Mini Actions and Theme Integration

**Files:**
- Modify: `ui.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `navigation_frame` and `nav` from Task 3.
- Produces: `navigation_actions`, with existing `reconnect_button`, `test_button`, and `theme_button` children; `_nav_palette() -> dict[str, str]`.

- [ ] **Step 1: Write failing ownership and visibility tests**

```python
def test_navigation_keeps_three_mini_actions_visible(self):
    for button in (self.app.reconnect_button, self.app.test_button,
                   self.app.theme_button):
        self.assertIs(button.master, self.app.navigation_actions)
    self.assertEqual(
        [b.cget("text") for b in (self.app.reconnect_button,
                                  self.app.test_button,
                                  self.app.theme_button)],
        ["↻", "▶", "☾"],
    )

def test_mini_actions_remain_visible_on_every_page(self):
    self.app.deiconify()
    for index in range(3):
        self.app.select_page(index)
        self.app.update_idletasks()
        self.assertTrue(all(b.winfo_ismapped() for b in (
            self.app.reconnect_button, self.app.test_button,
            self.app.theme_button)))
```

- [ ] **Step 2: Run the UI suite and verify RED**

Run: `python -m unittest discover -s tests -p test_ui.py -v`

Expected: ownership failures because actions still live in Tools/Footer.

- [ ] **Step 3: Move the existing buttons without changing commands**

Create `navigation_actions` on the right of `navigation_frame`. Build all three
buttons there with `width=3`, existing commands, disabled-state updates, and
tooltips. Remove the Tools card and theme button from the footer. Update
`toggle_theme()` to call `self.nav.set_palette(self._nav_palette())`.

```python
def _nav_palette(self) -> dict[str, str]:
    p = self._palette
    return {
        "background": p["window"], "capsule": p["secondary"],
        "capsule_outline": p["border"], "pill": p["primary"],
        "pill_highlight": "#FFFFFF", "text": p["text"],
        "active_text": "#FFFFFF", "focus": p["focus"],
    }
```

- [ ] **Step 4: Run `python -m unittest discover -s tests -p test_ui.py -v` and verify GREEN**

Expected: layout and runtime tests pass, including reconnect, Test Run disabled
state, theme persistence, and tooltips.

- [ ] **Step 5: Commit**

```powershell
git add -- ui.py tests/test_ui.py
git commit -m "feat: move mini actions into navigation"
```

---

### Task 5: Advanced Scrolling, STOP Visibility, and Shutdown

**Files:**
- Modify: `ui.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `nav.cancel_animation()`, `select_page`, and the Advanced canvas.
- Produces: `_on_advanced_mousewheel(event) -> str | None` and teardown that cancels nav animation before `destroy()`.

- [ ] **Step 1: Write failing integration tests**

```python
def test_stop_is_visible_on_every_navigation_page(self):
    self.app.deiconify()
    for index in range(3):
        self.app.select_page(index)
        self.app.update()
        self.assertEqual(self.app.stop_button.winfo_ismapped(), 1)

def test_close_cancels_navigation_animation(self):
    self.app.nav.select(2)
    self.assertIsNotNone(self.app.nav._animation_after_id)
    self.app.close_app()
    self.assertIsNone(self.app.nav._animation_after_id)

def test_advanced_canvas_belongs_only_to_advanced_page(self):
    self.assertTrue(self._is_descendant(self.app.advanced_canvas,
                                        self.app.advanced_page))
    self.assertFalse(self._is_descendant(self.app.stop_button,
                                         self.app.advanced_page))

def test_invalid_advanced_edit_does_not_change_page(self):
    self.app.select_page(2)
    self.app.horizontal_jitter_pps_var.set("not-a-number")
    self.app._motion_changed("horizontal_jitter_pps")
    self.assertEqual(self.app.nav.selected_index, 2)
    self.assertTrue(self.app.footer_var.get().startswith("Invalid value for "))
```

- [ ] **Step 2: Run the UI suite and verify RED**

Run: `python -m unittest discover -s tests -p test_ui.py -v`

Expected: failures for missing page-scoped canvas name or nav cancellation.

- [ ] **Step 3: Finish scoped scrolling and teardown**

Rename the old right-workspace wheel handler to `_on_advanced_mousewheel`. It
returns immediately unless Advanced is selected and the pointer is inside its
canvas host. Keep STOP/runtime at shell row 4. In `close_app()`, immediately
after setting `_closing`, call `self.nav.cancel_animation()` before service
cleanup or `destroy()`.

- [ ] **Step 4: Run widget and UI suites and verify GREEN**

```powershell
python -m unittest discover -s tests -p test_xp_widgets.py -v
python -m unittest discover -s tests -p test_ui.py -v
```

Expected: both suites pass without Tcl callback errors.

- [ ] **Step 5: Commit**

```powershell
git add -- ui.py xp_widgets.py tests/test_ui.py tests/test_xp_widgets.py
git commit -m "fix: keep Liquid XP navigation shutdown-safe"
```

---

### Task 6: Full Verification

**Files:**
- Modify only if verification exposes a scoped defect: `ui.py`, `xp_widgets.py`, `tests/test_ui.py`, `tests/test_xp_widgets.py`

**Interfaces:**
- Consumes: completed Liquid XP navigation and unchanged runtime interfaces.
- Produces: verified source tree; no packaged executable.

- [ ] **Step 1: Run syntax verification**

Run: `python -m py_compile main.py ui.py motion.py makcu_service.py hotkeys.py settings.py`

Expected: exit code 0 with no output.

- [ ] **Step 2: Run `python -m unittest discover -s tests -v`**

Expected: all tests pass with no errors, failures, or Tcl warnings.

- [ ] **Step 3: Run `python -c "import makcu"`**

Expected: exit code 0 with no output.

- [ ] **Step 4: Inspect final boundaries**

Run `git diff --check`, then `git status --short`.

Expected: no whitespace errors and no edits under `build-output/`, `dist/`,
`*.build/`, `*.dist/`, or `__pycache__/`.

- [ ] **Step 5: Commit a verification correction only if required**

Stage only the exact corrected source/test files and use a message naming the
defect. If no correction was needed, do not create an empty commit.
