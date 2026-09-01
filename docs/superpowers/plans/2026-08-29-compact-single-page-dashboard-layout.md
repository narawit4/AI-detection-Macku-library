# Compact Single-Page Dashboard Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Jitter's rail-and-pages UI with a compact fixed-size single-page dashboard containing five independently collapsible categories while preserving every runtime behavior.

**Architecture:** Add a presentation-only `CollapsibleSection` widget, then rebuild `JitterApp` around one shared scroll canvas with fixed top, footer, and runtime regions. Existing Tk variables, callbacks, services, control attributes, and safety paths remain authoritative; section summaries are derived on the Tk thread and never persisted.

**Tech Stack:** Python 3, standard-library Tkinter/ttk, `unittest`, existing Jitter presentation widgets

**Spec:** `docs/superpowers/specs/2026-08-29-compact-single-page-dashboard-layout-design.md`

## Global Constraints

- Keep the fixed English `840x620` Windows Tkinter window and `resizable(False, False)`.
- Use exactly five ordered sections: `Control`, `Jitter`, `AI Aim`, `Overlay`, and `Settings`.
- Start with only `Control` expanded; allow any number of sections to remain expanded.
- Use one content scrollbar and no nested scrolling regions.
- Keep the top bar, one-line footer, and `Master / Runtime / STOP` dock outside the scroll canvas.
- Keep `STOP` visible under every section and scroll state.
- Preserve a slider and exact-value entry for every numeric setting.
- Keep AI runtime metrics out of the main UI; FPS, provider, zoom, and lock remain in the overlay HUD.
- Place the Theme action only in `Settings`.
- Do not persist section state or add a configuration schema change.
- Do not change motion, AI, model selection, overlay, Makcu, hotkey, sound, cancellation, or shutdown behavior.
- Keep every Tk widget and Tk-variable access on the main thread.
- Add no dependency, model, asset, download, profile, tray behavior, or packaging work.
- Preserve the untracked experimental ONNX files without staging, copying, deleting, or modifying them.
- Follow red-green TDD and commit each reviewed task separately.
- Do not run Nuitka or `gen.bat`.

## File Map

- Modify `jitter_app/presentation/widgets.py`: add the reusable, behavior-free `CollapsibleSection` presentation primitive; retain existing widget classes.
- Modify `tests/test_liquid_widgets.py`: cover initial state, independent expansion, keyboard input, summary updates, callbacks, and trace cleanup.
- Modify `jitter_app/presentation/ui.py`: construct the single-page shell, rehome all controls, provide summaries, simplify action presentation, and replace page-local scrolling.
- Modify `tests/test_ui.py`: replace obsolete rail/page geometry assertions with section ownership, fixed chrome, scrolling, summaries, density, theme, persistence, and regression assertions.

No motion, AI, device, configuration, overlay-rendering, sound-service, packaging, requirement, model, or image file changes belong in this plan.

---

### Task 1: Add the Accessible Collapsible Section Primitive

**Files:**

- Modify: `jitter_app/presentation/widgets.py` near the palette constants and before `LiquidNavigation`
- Modify: `tests/test_liquid_widgets.py` imports and add `CollapsibleSectionTests`

**Interfaces:**

- Consumes: Tkinter/ttk, a caller-owned `tk.StringVar`, and existing ttk style names.
- Produces: `CollapsibleSection(parent, *, number: int, title: str, summary: tk.StringVar, expanded: bool = False, on_toggle: Callable[[bool], None] | None = None)`.
- Produces attributes: `header_button: ttk.Button`, `body: ttk.Frame`, and read-only `expanded: bool`.
- Produces methods: `set_expanded(expanded: bool, *, notify: bool = True) -> None` and `toggle() -> None`.

- [ ] **Step 1: Write failing widget tests**

Import the new class and add this fixture and contract:

```python
from jitter_app.presentation.widgets import (
    CollapsibleSection,
    LiquidIconButton,
    LiquidNavigation,
    LiquidSlider,
)


class CollapsibleSectionTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.summary = tk.StringVar(self.root, "Ready")
        self.events = []
        self.section = CollapsibleSection(
            self.root,
            number=1,
            title="Control",
            summary=self.summary,
            expanded=True,
            on_toggle=self.events.append,
        )
        self.section.pack(fill="x")
        self.root.update_idletasks()

    def tearDown(self):
        destroy_tk_test_root(self, "section")

    def test_initial_state_exposes_one_focusable_full_width_header(self):
        self.assertTrue(self.section.expanded)
        self.assertIsInstance(self.section.header_button, ttk.Button)
        self.assertEqual(str(self.section.header_button.cget("takefocus")), "1")
        self.assertEqual(self.section.body.winfo_manager(), "grid")
        self.assertIn("01", self.section.header_button.cget("text"))
        self.assertIn("CONTROL", self.section.header_button.cget("text"))
        self.assertIn("Ready", self.section.header_button.cget("text"))

    def test_set_expanded_is_idempotent_and_notifies_only_on_change(self):
        self.section.set_expanded(True)
        self.assertEqual(self.events, [])
        self.section.set_expanded(False)
        self.assertFalse(self.section.expanded)
        self.assertEqual(self.section.body.winfo_manager(), "")
        self.assertEqual(self.events, [False])
        self.section.set_expanded(False)
        self.assertEqual(self.events, [False])

    def test_return_and_space_toggle_exactly_once(self):
        self.root.deiconify()
        self.root.update()
        self.section.header_button.focus_force()
        self.section.header_button.event_generate("<Return>")
        self.root.update()
        self.assertFalse(self.section.expanded)
        self.section.header_button.event_generate("<space>")
        self.root.update()
        self.assertTrue(self.section.expanded)
        self.assertEqual(self.events, [False, True])

    def test_summary_variable_refreshes_header_without_changing_state(self):
        self.summary.set("Jitter | Left | Balanced")
        self.root.update_idletasks()
        self.assertIn(
            "Jitter | Left | Balanced",
            self.section.header_button.cget("text"),
        )
        self.assertTrue(self.section.expanded)

    def test_two_sections_expand_independently(self):
        second = CollapsibleSection(
            self.root,
            number=2,
            title="Jitter",
            summary=tk.StringVar(self.root, "2 px | 60 Hz | Smooth"),
        )
        second.pack(fill="x")
        second.set_expanded(True)
        self.assertTrue(self.section.expanded)
        self.assertTrue(second.expanded)

    def test_destroy_removes_the_summary_trace(self):
        before = len(self.summary.trace_info())
        self.section.destroy()
        self.root.update_idletasks()
        self.assertEqual(len(self.summary.trace_info()), before - 1)
        self.section = None
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest tests.test_liquid_widgets.CollapsibleSectionTests -v
```

Expected: import failure because `CollapsibleSection` does not exist.

- [ ] **Step 3: Implement the minimal widget**

Add the class without application-specific reads or callbacks:

```python
class CollapsibleSection(ttk.Frame):
    """One independently collapsible, keyboard-accessible dashboard section."""

    def __init__(
        self,
        parent,
        *,
        number: int,
        title: str,
        summary: tk.StringVar,
        expanded: bool = False,
        on_toggle: Callable[[bool], None] | None = None,
    ) -> None:
        if int(number) < 1:
            raise ValueError("section number must be positive")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("section title must be non-empty")
        super().__init__(parent, style="Liquid.SettingsCard.TFrame")
        self.number = int(number)
        self.title = title.strip().upper()
        self.summary = summary
        self.on_toggle = on_toggle
        self._expanded = not bool(expanded)
        self._summary_trace_id = self.summary.trace_add(
            "write", self._summary_changed
        )

        self.columnconfigure(0, weight=1)
        self.header_button = ttk.Button(
            self,
            text="",
            command=self.toggle,
            takefocus=True,
            style="Liquid.Section.TButton",
        )
        self.header_button.grid(row=0, column=0, sticky="ew")
        self.header_button.bind("<Return>", self._key_toggle)
        self.header_button.bind("<space>", self._key_toggle)
        self.body = ttk.Frame(
            self,
            style="Liquid.SectionBody.TFrame",
            padding=(10, 9),
        )
        self.body.grid(row=1, column=0, sticky="ew")
        self.bind("<Destroy>", self._destroyed, add="+")
        self.set_expanded(expanded, notify=False)

    @property
    def expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool, *, notify: bool = True) -> None:
        expanded = bool(expanded)
        if expanded == self._expanded:
            return
        self._expanded = expanded
        if expanded:
            self.body.grid()
        else:
            self.body.grid_remove()
        self._refresh_header()
        if notify and self.on_toggle is not None:
            self.on_toggle(expanded)

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def _key_toggle(self, _event=None) -> str:
        self.toggle()
        return "break"

    def _summary_changed(self, *_args) -> None:
        self._refresh_header()

    def _refresh_header(self) -> None:
        marker = "\u25b2" if self._expanded else "\u25bc"
        summary = " ".join(str(self.summary.get()).split())
        self.header_button.configure(
            text=f"{self.number:02d}   {self.title}   {summary}   {marker}"
        )

    def _destroyed(self, event) -> None:
        if event.widget is not self or self._summary_trace_id is None:
            return
        trace_id = self._summary_trace_id
        self._summary_trace_id = None
        try:
            self.summary.trace_remove("write", trace_id)
        except tk.TclError:
            pass
```

- [ ] **Step 4: Run widget tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_liquid_widgets.CollapsibleSectionTests -v
python -m unittest discover -s tests -p test_liquid_widgets.py -v
```

Expected: both commands exit `0`; all existing slider, navigation, and icon-button tests remain green.

- [ ] **Step 5: Commit the primitive**

```powershell
git add jitter_app/presentation/widgets.py tests/test_liquid_widgets.py
git commit -m "feat: add collapsible dashboard sections"
```

### Task 2: Build the Single-Page Shell and Rehome Every Control

**Files:**

- Modify: `jitter_app/presentation/ui.py:52-4920`
- Modify: `tests/test_ui.py:1-2785`

**Interfaces:**

- Consumes: Task 1 `CollapsibleSection`.
- Produces shell attributes: `topbar_frame`, `dashboard_frame`, `dashboard_scroll_canvas`, `dashboard_scrollbar`, `dashboard_content`, `_dashboard_scroll_window`, `footer_frame`, and `runtime_frame`.
- Produces section attributes: `control_section`, `jitter_section`, `ai_section`, `overlay_section`, `settings_section`, and `sections: tuple[CollapsibleSection, ...]`.
- Preserves: all control attributes used by runtime code, `control_frame`, `quick_frame`, Tk variables, service callbacks, and command methods.

- [ ] **Step 1: Replace obsolete rail/page tests with failing single-page tests**

Import `CollapsibleSection` in `tests/test_ui.py` and replace the tests that require `navigation_rail`, `nav`, `pages`, page titles, page selection, or page-local scrolling with these contracts:

```python
from jitter_app.presentation.widgets import CollapsibleSection, LiquidSlider


def test_single_page_shell_orders_fixed_chrome_around_one_scroll_region(self):
    widgets = (
        self.app.topbar_frame,
        self.app.dashboard_frame,
        self.app.footer_frame,
        self.app.runtime_frame,
    )
    self.assertEqual(
        [int(widget.grid_info()["row"]) for widget in widgets],
        [0, 1, 2, 3],
    )
    self.assertTrue(all(widget.master is self.app.console_workspace for widget in widgets))
    self.assertIsInstance(self.app.dashboard_scroll_canvas, tk.Canvas)
    self.assertFalse(hasattr(self.app, "navigation_rail"))
    self.assertFalse(hasattr(self.app, "nav"))
    self.assertFalse(hasattr(self.app, "pages"))


def test_dashboard_has_five_ordered_independent_sections(self):
    self.assertEqual(
        tuple(section.title for section in self.app.sections),
        ("CONTROL", "JITTER", "AI AIM", "OVERLAY", "SETTINGS"),
    )
    self.assertTrue(all(isinstance(section, CollapsibleSection)
                        for section in self.app.sections))
    self.assertTrue(self.app.control_section.expanded)
    self.assertTrue(all(not section.expanded for section in self.app.sections[1:]))
    self.app.ai_section.set_expanded(True)
    self.app.overlay_section.set_expanded(True)
    self.assertTrue(self.app.control_section.expanded)
    self.assertTrue(self.app.ai_section.expanded)
    self.assertTrue(self.app.overlay_section.expanded)


def test_dashboard_uses_one_vertical_scrollbar_and_no_motion_scroll(self):
    vertical = [
        widget
        for widget in descendant_widgets(self.app.console_workspace)
        if isinstance(widget, ttk.Scrollbar)
        and str(widget.cget("orient")) == "vertical"
    ]
    self.assertEqual(vertical, [self.app.dashboard_scrollbar])
    self.assertFalse(hasattr(self.app, "motion_scroll_canvas"))
    self.assertFalse(hasattr(self.app, "motion_scrollbar"))


def test_fixed_runtime_dock_keeps_stop_visible_at_bottom_of_dashboard(self):
    self.app.deiconify()
    for section in self.app.sections:
        section.set_expanded(True)
    self.app.update()
    self.app.dashboard_scroll_canvas.yview_moveto(1.0)
    self.app.update()
    self.assertTrue(self.app.footer_frame.winfo_ismapped())
    self.assertTrue(self.app.runtime_frame.winfo_ismapped())
    self.assertTrue(self.app.stop_button.winfo_ismapped())
    self.assertIs(self.app.stop_button.master, self.app.runtime_frame)
    self.assertLessEqual(
        self.app.stop_button.winfo_rooty() + self.app.stop_button.winfo_height(),
        self.app.winfo_rooty() + self.app.winfo_height(),
    )
```

Add one ownership test covering every category:

```python
def test_each_existing_control_belongs_to_its_approved_section(self):
    ownership = {
        self.app.control_section.body: (
            self.app.jitter_source_button, self.app.ai_source_button,
            self.app.trigger_combo, self.app.modifier_combo,
            self.app.hotkey_button, self.app.preset_combo,
            self.app.device_label, self.app.reconnect_button,
            self.app.test_button,
        ),
        self.app.jitter_section.body: (
            self.app.pulse_size_px_scale, self.app.pulse_size_px_entry,
            self.app.pulse_rate_hz_scale, self.app.pulse_rate_hz_entry,
            self.app.ramp_mode_combo,
        ),
        self.app.ai_section.body: (
            self.app.ai_confidence_scale, self.app.ai_confidence_entry,
            self.app.ai_aim_strength_scale, self.app.ai_aim_strength_entry,
            self.app.ai_smoothing_scale, self.app.ai_smoothing_entry,
            self.app.ai_max_step_scale, self.app.ai_max_step_entry,
            self.app.target_area_combo, self.app.model_browse_button,
            self.app.use_default_model_button, self.app.ai_curve_canvas,
            self.app.ai_curve_reset_button,
        ),
        self.app.overlay_section.body: (
            self.app.overlay_button, self.app.overlay_reset_button,
            self.app.overlay_color_button, self.app.overlay_head_button,
            self.app.overlay_player_button, self.app.overlay_box_width_scale,
            self.app.overlay_box_width_entry, self.app.overlay_label_mode_combo,
            self.app.overlay_hud_button, self.app.overlay_hud_color_button,
            self.app.overlay_hud_corner_combo,
            self.app.overlay_hud_offset_x_scale,
            self.app.overlay_hud_offset_y_scale,
            self.app.overlay_hud_font_size_scale,
        ),
        self.app.settings_section.body: (
            self.app.sound_enabled_check, self.app.sound_volume_scale,
            self.app.sound_volume_entry, self.app.test_on_button,
            self.app.test_off_button, self.app.theme_button,
        ),
    }
    for section_body, widgets in ownership.items():
        for widget in widgets:
            with self.subTest(section=section_body, widget=widget):
                self.assertTrue(self._is_descendant(widget, section_body))
```

Retain service, configuration, movement, AI, overlay, sound, STOP, shutdown, contrast, and slider behavior tests. Change only their obsolete layout setup:

- replace each old `select_page(index)` call with `set_expanded(True)` on the relevant section;
- replace `motion_scroll_canvas` with `dashboard_scroll_canvas`;
- replace `motion_scroll_content` with the relevant section body;
- replace page ancestry with section-body ancestry;
- remove assertions for `navigation_actions`, page headers, the rail width, and `LiquidNavigation` ownership.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_ui.JitterLayoutTests.test_single_page_shell_orders_fixed_chrome_around_one_scroll_region -v
python -m unittest tests.test_ui.JitterLayoutTests.test_dashboard_has_five_ordered_independent_sections -v
python -m unittest tests.test_ui.JitterLayoutTests.test_each_existing_control_belongs_to_its_approved_section -v
```

Expected: failures for missing dashboard and section attributes.

- [ ] **Step 3: Register compact section styles**

Import the new widget and create section-specific ttk styles in `_configure_styles`:

```python
from .widgets import CollapsibleSection, LiquidIconButton, LiquidSlider

# Inside _configure_styles, beside the other rounded elements:
section_element = self._install_rounded_element(
    style,
    "Section",
    (
        p["surface"], p["raised"], p["surface"],
        disabled_background, p["border"],
    ),
    focus=(p["surface"], p["accent"]),
)
style.configure(
    "Liquid.Section.TButton",
    background=p["surface"],
    foreground=p["text"],
    bordercolor=p["border"],
    font=(FONT_FAMILY, 10, "bold"),
    padding=(10, 8),
    anchor="w",
)
style.map(
    "Liquid.Section.TButton",
    background=[("pressed", p["surface"]), ("active", p["raised"])],
    foreground=[("disabled", disabled_text)],
)
style.configure(
    "Liquid.SectionBody.TFrame",
    background=p["surface"],
    bordercolor=p["border"],
    relief="flat",
    borderwidth=0,
)
style.layout("Liquid.Section.TButton", button_layout(section_element))
```

Move the `button_layout` helper above the first call that uses it so the new style and existing button styles share the same layout function.

- [ ] **Step 4: Replace `_build_page` with the four-region shell**

Build one workspace column and create the fixed regions in this exact order:

```python
def _build_page(self) -> None:
    self.shell = tk.Canvas(
        self,
        background=self._palette["window"],
        highlightthickness=0,
        borderwidth=0,
        takefocus=False,
    )
    self.shell.pack(fill="both", expand=True)
    self.shell.columnconfigure(0, weight=1)
    self.shell.rowconfigure(0, weight=1)
    self.shell.bind("<Configure>", self._redraw_shell_art, add="+")

    self.console_workspace = ttk.Frame(
        self.shell, style="Liquid.App.TFrame", padding=(12, 10)
    )
    self.console_workspace.grid(row=0, column=0, sticky="nsew")
    self.console_workspace.columnconfigure(0, weight=1)
    self.console_workspace.rowconfigure(1, weight=1)

    self._build_topbar()
    self._build_dashboard()
    self._build_footer()
    self._build_main_control_card()
    self._apply_combobox_popup_palette()
    self._redraw_shell_art()
```

Grid `topbar_frame`, `dashboard_frame`, `footer_frame`, and `runtime_frame` at rows `0`, `1`, `2`, and `3`. Keep the existing three-column runtime weights and button commands. Reduce its padding to the approved compact range without changing styles or meaning.

Build the top bar with `identity_frame` at the left and the existing `connection_indicator` plus `connection_label` at the right. Show one `JITTER` identity label and no page title, subtitle, theme action, AI status, Reconnect, or Test action.

- [ ] **Step 5: Build the shared dashboard canvas and five sections**

Before constructing the sections, add these caller-owned variables to
`_create_variables`; Task 3 fills them with live content:

```python
self.control_section_summary_var = tk.StringVar(self, "No sources")
self.ai_section_summary_var = tk.StringVar(self, "Default model")
self.overlay_section_summary_var = tk.StringVar(self, "Overlay Off")
self.settings_section_summary_var = tk.StringVar(self, "Sound On")
```

Create the scroll boundary and sections with stable attributes:

```python
def _build_dashboard(self) -> None:
    self.dashboard_frame = ttk.Frame(
        self.console_workspace, style="Liquid.App.TFrame"
    )
    self.dashboard_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 6))
    self.dashboard_frame.columnconfigure(0, weight=1)
    self.dashboard_frame.rowconfigure(0, weight=1)
    self.dashboard_scroll_canvas = tk.Canvas(
        self.dashboard_frame,
        background=self._palette["window"],
        highlightthickness=0,
        borderwidth=0,
        takefocus=False,
    )
    self.dashboard_scroll_canvas.grid(row=0, column=0, sticky="nsew")
    self.dashboard_scrollbar = ttk.Scrollbar(
        self.dashboard_frame,
        orient="vertical",
        style="Liquid.Vertical.TScrollbar",
        command=self.dashboard_scroll_canvas.yview,
    )
    self.dashboard_scrollbar.grid(row=0, column=1, sticky="ns", padx=(6, 0))
    self.dashboard_scroll_canvas.configure(
        yscrollcommand=self.dashboard_scrollbar.set
    )
    self.dashboard_content = ttk.Frame(
        self.dashboard_scroll_canvas, style="Liquid.App.TFrame"
    )
    self.dashboard_content.columnconfigure(0, weight=1)
    self._dashboard_scroll_window = self.dashboard_scroll_canvas.create_window(
        (0, 0), window=self.dashboard_content, anchor="nw"
    )

    definitions = (
        ("control_section", 1, "Control", self.control_section_summary_var, True),
        ("jitter_section", 2, "Jitter", self.motion_summary_var, False),
        ("ai_section", 3, "AI Aim", self.ai_section_summary_var, False),
        ("overlay_section", 4, "Overlay", self.overlay_section_summary_var, False),
        ("settings_section", 5, "Settings", self.settings_section_summary_var, False),
    )
    sections = []
    for row, (attribute, number, title, summary, expanded) in enumerate(definitions):
        section = CollapsibleSection(
            self.dashboard_content,
            number=number,
            title=title,
            summary=summary,
            expanded=expanded,
        )
        section.grid(row=row, column=0, sticky="ew", pady=(0, 7))
        setattr(self, attribute, section)
        sections.append(section)
    self.sections = tuple(sections)

    self._build_control_section(self.control_section.body)
    self._build_jitter_section(self.jitter_section.body)
    self._build_ai_section(self.ai_section.body)
    self._build_overlay_section(self.overlay_section.body)
    self._build_settings_section(self.settings_section.body)
```

Bind content `<Configure>`, canvas `<Configure>`, and `<MouseWheel>` to renamed `_refresh_dashboard_scrollregion`, `_resize_dashboard_content`, and `_scroll_dashboard` methods. Width changes update `_dashboard_scroll_window`. Expanding or collapsing a body changes `dashboard_content` geometry, whose `<Configure>` binding refreshes the scrollregion without scheduling an unowned callback.

- [ ] **Step 6: Establish the builder contract and rehome Control**

Rename and split the current builders to these exact definitions:

- `_build_control_section(self, parent: ttk.Frame) -> None`
- `_build_jitter_section(self, parent: ttk.Frame) -> None`
- `_build_ai_section(self, parent: ttk.Frame) -> None`
- `_build_ai_curve_card(self, parent: ttk.Frame) -> None`
- `_build_overlay_section(self, parent: ttk.Frame) -> None`
- `_build_settings_section(self, parent: ttk.Frame) -> None`

Use the following complete ownership map while preserving each existing widget attribute and command:

| Builder | Required controls |
| --- | --- |
| `_build_control_section` | source buttons, Trigger, Modifier, Hotkey, Preset, device status, Reconnect, Test 3s |
| `_build_jitter_section` | Pulse Size slider/entry, Pulse Rate slider/entry, Ramp Mode |
| `_build_ai_section` | four AI slider/entry pairs, Target Area, current model label, Browse, Use Default, response curve, four exact curve entries, fixed zero, Reset Curve |
| `_build_overlay_section` | Overlay/Reset, box color, Head/Player, width slider/entry, label mode, HUD toggle/color/corner, X/Y/font slider/entries, four HUD metric buttons |
| `_build_settings_section` | sound enabled, Volume slider/entry, both cue previews, Theme |

Implement `_build_control_section` first. Set `self.control_frame = parent`, configure internal columns with weights `3:2`, and grid `control_bindings_card` and `control_device_card` side by side. Move the current source buttons, Trigger, Modifier, Hotkey, Preset, and bounded device label without changing their variables or commands. Add an action row in `control_device_card` and move the existing Reconnect and Test controls into it.

- [ ] **Step 7: Rehome Jitter controls without a nested canvas**

Implement `_build_jitter_section`. Set `self.quick_frame = parent`; create `motion_hero_card` directly under it; grid Pulse Size and Pulse Rate as the existing two numeric controls and Ramp Mode beneath them. Do not create `motion_scroll_frame`, `motion_scroll_canvas`, `motion_scrollbar`, or `motion_scroll_content`. Keep the existing `motion_summary_var` for the section header; Task 3 removes the redundant large snapshot card.

- [ ] **Step 8: Rehome AI settings and response curve**

Implement `_build_ai_section` with `ai_settings_card` at row `0` and call `_build_ai_curve_card(parent)` for row `1`. Preserve the existing four `_ai_numeric_control` calls, Target Area dropdown, model label, Browse, Use Default, curve variables, node bindings, exact entries, and Reset Curve command. Change `_build_ai_curve_card` to parent its card to the supplied frame rather than `motion_scroll_content`.

- [ ] **Step 9: Rehome Overlay controls**

Implement `_build_overlay_section` by parenting `overlay_custom_card` to the supplied frame. Preserve the two internal columns: detection boxes on the left and HUD controls on the right. Keep every toggle, color action, dropdown, numeric range, exact entry, and HUD metric callback unchanged.

- [ ] **Step 10: Rehome Settings and its single Theme action**

Implement `_build_settings_section` with `settings_content = parent`, retaining `sound_feedback_card` and `sound_preview_card` as the internal `3:2` columns. Move the existing Theme `LiquidIconButton` into the Settings action column for this task boundary; do not create a top-bar copy. Keep sound enablement, Volume, and both preview callbacks unchanged.

Keep `control_bindings_card`, `control_device_card`, `motion_hero_card`, `ai_settings_card`, `ai_curve_card`, `overlay_custom_card`, `settings_content`, `sound_feedback_card`, and `sound_preview_card` as compact subcontainers so runtime and integration seams remain stable.

At this task boundary, existing `LiquidIconButton` instances may be reused inside the new Control and Settings action rows. Task 3 replaces their icon-only presentation with final labeled ttk buttons after all behavior call sites are covered.

- [ ] **Step 11: Replace page-aware theme and shell drawing**

Update `toggle_theme` to recolor `dashboard_scroll_canvas` instead of `motion_scroll_canvas`, remove `nav.set_palette`, and keep current palette application for the curve, connection indicator, combobox popups, and every slider.

Update `_redraw_shell_art` so background bands span the full shell and rounded surfaces correspond to `topbar_frame`, `dashboard_frame`, and `runtime_frame`. Preserve semantic tags `workspace-band`, `rounded-surface`, `floating-panel`, `floating-panel-topbar`, `floating-panel-dashboard`, and `floating-panel-runtime`; remove rail-only tags.

Delete visible page switching and do not replace `select_page` with hidden navigation. Expansion is owned exclusively by each `CollapsibleSection`.

Remove `self.nav.cancel_animation()` from `close_app`; there is no replacement navigation callback. Keep all existing AI-curve, slider, save, capture, queue, and overlay callback cancellation.

- [ ] **Step 12: Run the layout and complete UI suites**

Run:

```powershell
python -m unittest tests.test_ui.JitterLayoutTests.test_single_page_shell_orders_fixed_chrome_around_one_scroll_region -v
python -m unittest tests.test_ui.JitterLayoutTests.test_dashboard_has_five_ordered_independent_sections -v
python -m unittest tests.test_ui.JitterLayoutTests.test_dashboard_uses_one_vertical_scrollbar_and_no_motion_scroll -v
python -m unittest tests.test_ui.JitterLayoutTests.test_fixed_runtime_dock_keeps_stop_visible_at_bottom_of_dashboard -v
python -m unittest tests.test_ui.JitterLayoutTests.test_each_existing_control_belongs_to_its_approved_section -v
python -m unittest discover -s tests -p test_ui.py -v
```

Expected: all commands exit `0`. Existing behavioral tests pass with only their layout setup updated.

- [ ] **Step 13: Commit the single-page structure**

```powershell
git add jitter_app/presentation/ui.py tests/test_ui.py
git commit -m "feat: build compact single-page dashboard"
```

### Task 3: Add Live Summaries and Final Compact Treatments

**Files:**

- Modify: `jitter_app/presentation/ui.py`
- Modify: `tests/test_ui.py`

**Interfaces:**

- Consumes: Task 2 section attributes and existing validated snapshots.
- Consumes summary variables: `control_section_summary_var`, `ai_section_summary_var`, `overlay_section_summary_var`, and `settings_section_summary_var` created by Task 2; `motion_summary_var` remains the Jitter summary.
- Produces helper: `_compact_section_summary(*parts: object, limit: int = 72) -> str`.
- Produces method: `_refresh_section_summaries() -> None`, called only on the Tk thread.
- Produces method: `_set_section_summary(variable: tk.StringVar, *parts: object) -> None`, which contains summary-only Tk failures.
- Produces helper: `_set_test_button_enabled(enabled: bool) -> None` for final ttk Test-button state.

- [ ] **Step 1: Write failing summary and compact-action tests**

Add these tests:

```python
def test_collapsed_section_summaries_follow_validated_live_state(self):
    self.assertIn("No sources", self.app.control_section_summary_var.get())
    self.assertEqual(
        self.app.motion_summary_var.get(),
        "2 px paired pulse at 60 Hz | Smooth",
    )
    self.assertIn("Head", self.app.ai_section_summary_var.get())
    self.assertIn("Overlay Off", self.app.overlay_section_summary_var.get())
    self.assertIn("Sound On", self.app.settings_section_summary_var.get())

    self.app.toggle_jitter_source()
    self.app.pulse_size_px_var.set("4")
    self.app._motion_changed("pulse_size_px")
    self.app.overlay_player_visible = False
    self.app._render_runtime_controls()
    self.app.sound_volume_var.set("45")
    self.app.apply_sound_settings()

    self.assertIn("Jitter", self.app.control_section_summary_var.get())
    self.assertIn("4 px", self.app.jitter_section.summary.get())
    self.assertIn("Head", self.app.overlay_section_summary_var.get())
    self.assertNotIn("Player", self.app.overlay_section_summary_var.get())
    self.assertIn("45%", self.app.settings_section_summary_var.get())


def test_invalid_ai_text_does_not_replace_valid_ai_summary(self):
    before = self.app.ai_section_summary_var.get()
    self.app.ai_vars["aim_strength"].set("invalid")
    self.app._ai_changed("aim_strength")
    self.assertEqual(self.app.ai_section_summary_var.get(), before)


def test_main_dashboard_excludes_ai_runtime_readouts(self):
    visible = set(widget_texts(self.app.dashboard_content))
    self.assertNotIn("AI RUNTIME", visible)
    self.assertNotIn(self.app.ai_fps_var.get(), visible)
    self.assertNotIn(self.app.ai_provider_var.get(), visible)
    self.assertNotIn(self.app.ai_cadence_var.get(), visible)


def test_theme_action_exists_once_inside_settings(self):
    self.assertTrue(
        self._is_descendant(self.app.theme_button, self.app.settings_section.body)
    )
    self.assertFalse(self._is_descendant(self.app.theme_button, self.app.topbar_frame))
    self.assertEqual(self.app.theme_button.cget("text"), "Switch to Dark Mode")
    self.app.theme_button.invoke()
    self.assertEqual(self.app.theme_button.cget("text"), "Switch to Light Mode")
    self.assertIn("Dark", self.app.settings_section_summary_var.get())


def test_session_actions_and_sound_previews_use_compact_labels(self):
    self.assertEqual(self.app.reconnect_button.cget("text"), "Reconnect")
    self.assertEqual(self.app.test_button.cget("text"), "Test 3s")
    self.assertEqual(self.app.test_on_button.cget("text"), "Play Armed Cue")
    self.assertEqual(self.app.test_off_button.cget("text"), "Play Disabled Cue")


def test_numeric_controls_still_pair_slider_and_exact_entry(self):
    pairs = (
        (self.app.pulse_size_px_scale, self.app.pulse_size_px_entry),
        (self.app.pulse_rate_hz_scale, self.app.pulse_rate_hz_entry),
        (self.app.ai_confidence_scale, self.app.ai_confidence_entry),
        (self.app.ai_aim_strength_scale, self.app.ai_aim_strength_entry),
        (self.app.ai_smoothing_scale, self.app.ai_smoothing_entry),
        (self.app.ai_max_step_scale, self.app.ai_max_step_entry),
        (self.app.overlay_box_width_scale, self.app.overlay_box_width_entry),
        (self.app.overlay_hud_offset_x_scale, self.app.overlay_hud_offset_x_entry),
        (self.app.overlay_hud_offset_y_scale, self.app.overlay_hud_offset_y_entry),
        (self.app.overlay_hud_font_size_scale, self.app.overlay_hud_font_size_entry),
        (self.app.sound_volume_scale, self.app.sound_volume_entry),
    )
    for slider, entry in pairs:
        self.assertIsInstance(slider, LiquidSlider)
        self.assertIsInstance(entry, ttk.Entry)
        self.assertEqual(int(entry.cget("width")), 5)
```

- [ ] **Step 2: Run the summary tests and verify RED**

Run:

```powershell
python -m unittest tests.test_ui.JitterLayoutTests.test_collapsed_section_summaries_follow_validated_live_state -v
python -m unittest tests.test_ui.JitterLayoutTests.test_theme_action_exists_once_inside_settings -v
python -m unittest tests.test_ui.JitterLayoutTests.test_session_actions_and_sound_previews_use_compact_labels -v
```

Expected: failures for static placeholder summaries and final labeled-button presentation.

- [ ] **Step 3: Add summary formatting and live refresh**

Add the pure clamp helper near the existing presentation formatting helpers:

```python
def _compact_section_summary(*parts: object, limit: int = 72) -> str:
    text = " | ".join(
        " ".join(str(part).split())
        for part in parts
        if str(part).strip()
    )
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
```

Use the four `StringVar` instances created by Task 2 and call `_refresh_section_summaries()` once after all section content exists.

Contain presentation-only summary failures without changing application state:

```python
def _set_section_summary(
    self, variable: tk.StringVar, *parts: object
) -> None:
    try:
        variable.set(_compact_section_summary(*parts))
    except (tk.TclError, RuntimeError, TypeError, ValueError):
        logging.debug("Could not refresh dashboard summary", exc_info=True)
```

Use validated snapshots and display labels only:

```python
def _refresh_section_summaries(self) -> None:
    if self._closing:
        return
    source_names = [
        name
        for selected, name in (
            (self.jitter_selected, "Jitter"),
            (self.ai_selected, "AI Aim"),
        )
        if selected
    ]
    sources = " + ".join(source_names) or "No sources"
    self._set_section_summary(
        self.control_section_summary_var,
        sources,
        self.trigger_var.get(),
        self.preset_var.get(),
        self.connection_status_var.get(),
    )

    aim = self.get_ai_settings()
    target = _TARGET_AREA_LABELS.get(aim.target_area, "Head")
    self._set_section_summary(
        self.ai_section_summary_var,
        self.ai_model_var.get(),
        target,
        f"Strength {_display_value(aim.aim_strength)}",
    )

    box_names = [
        name
        for visible, name in (
            (self.overlay_head_visible, "Head"),
            (self.overlay_player_visible, "Player"),
        )
        if visible
    ]
    boxes = " + ".join(box_names) or "No boxes"
    hud = (
        f"HUD {self.overlay_hud_corner_var.get()}"
        if self.overlay_hud_visible else "HUD Off"
    )
    self._set_section_summary(
        self.overlay_section_summary_var,
        f"Overlay {'On' if self.overlay_visible else 'Off'}",
        boxes,
        hud,
    )

    try:
        volume = max(0, min(100, int(self.sound_volume_var.get())))
    except (TypeError, ValueError):
        volume = self.config.sound_volume
    self._set_section_summary(
        self.settings_section_summary_var,
        f"Sound {'On' if self.sound_enabled_var.get() else 'Off'}",
        f"{volume}%",
        self.theme_var.get().title(),
    )
```

Invoke this method after `_set_connection_state`, `_render_runtime_controls`, `_render_model_controls`, `_replace_ai_snapshot`, `_motion_changed`, `apply_preset`, `_overlay_style_changed`, `apply_sound_settings`, `on_bindings_changed`, and `toggle_theme` complete their existing work. Invalid scalar edits return before refreshing the validated AI summary. Keep `motion_summary_var` directly attached to `jitter_section` so `_replace_motion_snapshot` remains its single update path.

- [ ] **Step 4: Remove duplicated large presentation and finalize labeled actions**

Apply these compact treatments without deleting authoritative variables:

- remove page eyebrows, 22-point page titles, repeated subtitles, and long card descriptions;
- remove `motion_summary_card` and its large readout labels while retaining `motion_summary_var` and the existing snapshot variables for compatibility;
- reduce `ai_curve_canvas` from 176 to 132 nominal pixels while retaining plot bounds, node hit targets, exact entries, and Reset Curve;
- remove the 30-point Volume readout and keep `sound_volume_entry` next to its slider;
- build Reconnect and Test as `ttk.Button` controls with `Reconnect` and `Test 3s` labels in `Control`;
- build Theme as one `ttk.Button` in `Settings`, labeled `Switch to Dark Mode` or `Switch to Light Mode` from current state;
- label the cue-preview buttons `Play Armed Cue` and `Play Disabled Cue`;
- remove the mini-action and theme tooltip builders and their close-time cleanup because every final action is self-labeled.
- remove `LiquidIconButton` and `LiquidNavigation` from `ui.py` imports after their final application use is gone; retain both classes and their tests in `widgets.py`.

Replace icon-button enablement at every Test lifecycle transition with:

```python
def _set_test_button_enabled(self, enabled: bool) -> None:
    self.test_button.configure(state="normal" if enabled else "disabled")
```

Update `toggle_theme` with:

```python
self.theme_button.configure(
    text=(
        "Switch to Light Mode"
        if self._theme == "dark"
        else "Switch to Dark Mode"
    )
)
self._refresh_section_summaries()
```

Do not add a second Theme control to the top bar.

- [ ] **Step 5: Run compactness, behavior, and full UI tests**

Run:

```powershell
python -m unittest tests.test_ui.JitterLayoutTests.test_collapsed_section_summaries_follow_validated_live_state -v
python -m unittest tests.test_ui.JitterLayoutTests.test_invalid_ai_text_does_not_replace_valid_ai_summary -v
python -m unittest tests.test_ui.JitterLayoutTests.test_main_dashboard_excludes_ai_runtime_readouts -v
python -m unittest tests.test_ui.JitterLayoutTests.test_theme_action_exists_once_inside_settings -v
python -m unittest tests.test_ui.JitterLayoutTests.test_session_actions_and_sound_previews_use_compact_labels -v
python -m unittest tests.test_ui.JitterLayoutTests.test_numeric_controls_still_pair_slider_and_exact_entry -v
python -m unittest discover -s tests -p test_ui.py -v
```

Expected: all commands exit `0`; source selection, Test state, theme, sound preview, numeric validation, model selection, Overlay, and STOP tests remain green.

- [ ] **Step 6: Commit summaries and compact controls**

```powershell
git add jitter_app/presentation/ui.py tests/test_ui.py
git commit -m "feat: compact dashboard controls and summaries"
```

### Task 4: Harden Integration and Run Required Verification

**Files:**

- Modify: `tests/test_ui.py`

**Interfaces:**

- Consumes: completed single-page dashboard from Tasks 1-3.
- Produces: regression evidence for geometry, theme, expansion, persistence, teardown, and all existing runtime behavior.

- [ ] **Step 1: Add cross-state regression tests**

Add tests that preserve state while manipulating presentation only:

```python
def test_expansion_and_theme_changes_preserve_values_and_outer_geometry(self):
    expected_geometry = self.app.geometry()
    expected_motion = self.app.get_motion_settings()
    expected_ai = self.app.get_ai_settings()
    expected_bindings = (
        self.app.trigger_var.get(),
        self.app.modifier_var.get(),
        self.app.hotkey_name_var.get(),
    )
    for section in self.app.sections:
        section.set_expanded(True)
    self.app.toggle_theme()
    self.app.dashboard_scroll_canvas.yview_moveto(1.0)
    self.app.update_idletasks()
    self.assertEqual(self.app.geometry(), expected_geometry)
    self.assertEqual(self.app.get_motion_settings(), expected_motion)
    self.assertEqual(self.app.get_ai_settings(), expected_ai)
    self.assertEqual(
        (
            self.app.trigger_var.get(),
            self.app.modifier_var.get(),
            self.app.hotkey_name_var.get(),
        ),
        expected_bindings,
    )


def test_section_expansion_is_not_serialized(self):
    self.app.ai_section.set_expanded(True)
    self.app.settings_section.set_expanded(True)
    self.app._cancel_after("_save_after_id")
    self.app.save_config()
    saved = self.store.saved[-1]
    self.assertFalse(hasattr(saved, "section_state"))
    self.assertFalse(hasattr(saved, "expanded_sections"))


def test_all_sections_expanded_create_scroll_without_moving_fixed_controls(self):
    self.app.deiconify()
    for section in self.app.sections:
        section.set_expanded(True)
    self.app.update()
    scrollregion = tuple(
        float(value)
        for value in self.app.tk.splitlist(
            self.app.dashboard_scroll_canvas.cget("scrollregion")
        )
    )
    self.assertGreater(scrollregion[3] - scrollregion[1],
                       self.app.dashboard_scroll_canvas.winfo_height())
    self.assertTrue(self.app.footer_frame.winfo_ismapped())
    self.assertTrue(self.app.stop_button.winfo_ismapped())
```

- [ ] **Step 2: Run the new integration tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_ui.JitterLayoutTests.test_expansion_and_theme_changes_preserve_values_and_outer_geometry -v
python -m unittest tests.test_ui.JitterLayoutTests.test_section_expansion_is_not_serialized -v
python -m unittest tests.test_ui.JitterLayoutTests.test_all_sections_expanded_create_scroll_without_moving_fixed_controls -v
```

Expected: all three tests pass because Tasks 1-3 already implement these contracts. On failure, do not commit Task 4; return to the named Task 2 or Task 3 red-green cycle that owns the failed contract.

- [ ] **Step 3: Prove obsolete page and rail code is gone from the app boundary**

Run:

```powershell
rg -n "navigation_rail|navigation_actions|select_page|motion_scroll_canvas|motion_scroll_content|control_page|motion_page|settings_page|motion_summary_card" jitter_app/presentation/ui.py tests/test_ui.py
```

Expected: no matches. `LiquidNavigation` and `LiquidIconButton` remain in `widgets.py` and `tests/test_liquid_widgets.py` for compatibility, but `JitterApp` no longer imports or instantiates them.

- [ ] **Step 4: Run focused and complete automated verification**

Run:

```powershell
$jitterSources = @('main.py', 'distribution_metadata.py') + @(Get-ChildItem -LiteralPath 'jitter_app' -Recurse -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
python -m py_compile @jitterSources
python -m unittest discover -s tests -v
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
git diff --check
```

Expected: every command exits `0`; the unittest run ends with `OK`; the AI self-check reports a successful bundled-model DirectML runtime; review JSON is emitted successfully; the diff check is empty.

- [ ] **Step 5: Commit integration tests**

```powershell
git add tests/test_ui.py
git commit -m "test: harden compact dashboard integration"
```

- [ ] **Step 6: Record the physical verification boundary**

Report that no Nuitka build was run. State that connection, Trigger/Modifier input, each Jitter/AI Aim source combination, combined movement, reconnect, Test 3s, global hotkey, STOP, disconnect, and hardware shutdown still require a connected Makcu device for physical verification.
