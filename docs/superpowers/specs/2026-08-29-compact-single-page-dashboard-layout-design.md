# Compact Single-Page Dashboard Layout Design

**Date:** 2026-08-29
**Status:** Approved for implementation
**Scope:** Presentation-only redesign of the fixed-size Tkinter application

## Summary

Replace the current 176-pixel navigation rail and three separate dashboard
pages with one full-width, vertically scrollable dashboard. The dashboard has
five independently collapsible sections: `Control`, `Jitter`, `AI Aim`,
`Overlay`, and `Settings`.

The redesign makes the application materially denser without changing motion,
AI, overlay, device, configuration, or threading behavior. `Control` is the
only section expanded at startup. Users may expand any number of sections at
the same time. A compact top bar, one-line footer, and the persistent
`Master / Runtime / STOP` dock remain outside the scrolling region.

AI inference runtime details do not appear in the main UI. FPS, provider, zoom,
and lock remain available through the overlay HUD, while the bottom runtime
dock continues to report general application movement state.

## Context

The current Focused Dashboard uses a persistent 176-pixel left rail, large page
headers, descriptive copy, and multiple padded cards spread across `Control`,
`Motion`, and `Settings` pages. Those elements create strong hierarchy but use
too much of the fixed `840x620` window for navigation and repeated context.

The approved direction is the single-page option from the layout exploration:

- no navigation rail or page switching;
- full-width collapsible categories;
- compact summaries visible while categories are collapsed;
- one shared content scrollbar;
- emergency and runtime actions fixed in place.

## Goals

- Use the fixed window area efficiently while keeping controls easy to scan.
- Keep every existing user-facing capability available.
- Organize controls into five clear, task-oriented categories.
- Keep `STOP` visible under every scroll and expansion state.
- Preserve sliders plus exact-value entries for every numeric setting.
- Make the default view concise while allowing multiple categories to remain
  open during cross-category tuning.
- Preserve the existing visual palette, state colors, and light/dark themes.
- Keep the redesign isolated to presentation code and presentation tests.

## Non-goals

- Changing the `840x620` fixed window size or making it resizable.
- Changing motion math, AI inference, model selection, targeting, adaptive
  zoom, overlay rendering, Makcu behavior, hotkey behavior, sound behavior, or
  cancellation semantics.
- Adding settings, profiles, persistence, downloads, or new runtime status.
- Persisting section expansion state.
- Restoring AI runtime metrics to the main application UI.
- Building or packaging the application with Nuitka.

## Approved decisions

| Decision | Approved result |
| --- | --- |
| Overall layout | Single-page collapsible sections |
| Expansion behavior | Sections expand independently; multiple may be open |
| Startup expansion | `Control` open; all other sections closed |
| Categories | `Control`, `Jitter`, `AI Aim`, `Overlay`, `Settings` |
| Navigation | Remove the left rail and visible page navigation |
| Scrolling | One scrollbar for the complete section stack |
| Fixed controls | Top bar, footer, and runtime dock do not scroll |
| Theme control | One control inside `Settings`; no duplicate in the top bar |
| AI runtime status | Overlay HUD only; no main-UI AI runtime card |

## Screen structure

The nominal layout remains `840x620`:

```text
+--------------------------------------------------------------------------+
| JITTER                                      MAKCU: CONNECTED             |  top bar
+--------------------------------------------------------------------------+
| 01  CONTROL   Jitter | Left | Balanced                              [v] |
| +----------------------------------------------------------------------+ |
| | activation and inputs                 device, sources, and session    | |
| +----------------------------------------------------------------------+ |
| 02  JITTER    2 px | 60 Hz | Smooth                               [v] |
| 03  AI AIM    Default 320 | Head | Strength ...                    [v] |  one
| 04  OVERLAY   Off | Head/Player | HUD Top Left                     [v] |  scroll
| 05  SETTINGS  Sound On | 70% | Dark                               [v] |  region
+--------------------------------------------------------------------------+
| READY | concise actionable status or error                               |  footer
+--------------------------------------------------------------------------+
| [ ENABLE SELECTED ]       RUNTIME | DISABLED                 [ STOP ]     |  dock
+--------------------------------------------------------------------------+
```

Nominal density targets are approximately 42 pixels for the top bar, 38 pixels
for a collapsed section header, 24 pixels for the footer, and 53 pixels for the
runtime dock. These are layout targets rather than DPI-independent API
contracts. Internal padding should generally be 8-10 pixels and inter-section
spacing 6-8 pixels.

### Top bar

The top bar contains only:

- the `JITTER` identity at the left; and
- the existing Makcu connection state at the right, using green, amber, and red
  for connected, connecting, and disconnected.

It has no page title, subtitle, AI metrics, theme button, Reconnect button, or
Test button. Those actions belong to their relevant sections.

### Scrollable content

All five sections are full-width and vertically stacked. Expanded section
content uses compact internal grids, usually two columns. Full-width stacking
avoids uneven card heights and wasted blank space between unrelated sections.

The section stack owns the only content scrollbar. Individual sections,
response-curve controls, and cards do not create nested scroll regions.

### Footer and runtime dock

The existing footer becomes a fixed, single-line status strip immediately
above the runtime dock. It continues to show concise actionable messages and
errors. Detailed diagnostics continue to go only to `app.log`.

The runtime dock remains a three-column row:

- left: the existing enable/disable Master action;
- center: general movement state and a short supporting status;
- right: the existing red `STOP` action.

The dock never enters the scrolling canvas. `STOP` therefore remains visible
when any or all advanced sections are expanded and when the content is scrolled
to either end.

## Collapsible-section behavior

A reusable `CollapsibleSection` presentation widget owns each section header
and body.

- The complete header is one focusable button, not a small chevron-only target.
- Mouse click, `Enter`, and `Space` toggle that section.
- Expanding one section does not close another section.
- Startup state is deterministic: only `Control` is expanded.
- Expansion state exists only for the current process and is never serialized.
- Toggling a section does not mutate any application setting.
- Section content remains alive while collapsed. Collapsing changes geometry,
  not control values, callbacks, worker demand, or service state.
- Focus stays on the toggled header. The shared canvas updates its scrollregion
  after the geometry change.
- Mouse-wheel behavior is limited to the dashboard content region and uses the
  existing Windows-friendly scrolling convention.

Each header contains a two-digit category number, category name, concise live
summary, and chevron. The summary is presentation-only and is derived on the Tk
thread from existing validated Tk variables or immutable snapshots.

## Control ownership

### 01 - Control

`Control` contains the frequently used activation and session controls:

- Jitter and AI Aim independent source-selection buttons;
- Trigger dropdown;
- Modifier dropdown;
- global-hotkey capture button;
- motion Preset dropdown;
- Makcu status;
- Reconnect;
- `Test 3s`.

The section uses two compact internal groups: activation/input controls and
device/session controls. Existing rules for unselected sources, Master arming,
Test state, model-switch availability, connection state, and button enablement
remain unchanged.

The collapsed summary prioritizes selected sources, Trigger, Preset, and device
state. It must remain readable when no source is selected.

### 02 - Jitter

`Jitter` contains:

- Pulse Size slider and exact entry;
- Pulse Rate slider and exact entry;
- Ramp Mode dropdown.

The large `Live Snapshot` card is removed. Its useful values become the
single-line live section summary: pulse size, pulse rate, ramp mode, and the
active-profile indication when space permits. This removes duplicated large
readouts without removing information.

### 03 - AI Aim

`AI Aim` contains all non-runtime AI configuration:

- Confidence slider and exact entry;
- Aim Strength slider and exact entry;
- Smoothing slider and exact entry;
- Max Step slider and exact entry;
- Target Area dropdown;
- current model display;
- `Browse...` and `Use Default` actions;
- the five-point response-curve editor;
- exact curve entries at 25%, 50%, 75%, and 100% distance;
- fixed 0% value and `Reset Curve`.

The response curve remains the largest control. Its canvas may be shorter than
the current 176-pixel presentation if node positions, labels, hit targets, and
exact inputs remain clear and fully testable.

The collapsed summary uses the model display name, Target Area, and Aim
Strength. It must not show provider, inference FPS, display cadence, servo
cadence, zoom, or lock. External model paths remain runtime-only and are not
persisted.

### 04 - Overlay

`Overlay` contains every existing visual control:

- Overlay on/off and Reset Overlay;
- box color;
- Head Boxes and Player Boxes visibility;
- Box Width slider and exact entry;
- Box Label mode;
- HUD on/off and HUD color;
- HUD corner;
- HUD X Offset, HUD Y Offset, and HUD Font Size sliders plus exact entries;
- independent FPS, Provider, Zoom, and Lock metric filters.

Detection-box controls and HUD controls form the two internal columns. The
existing persistence boundary remains unchanged: only already approved schema
5 values are saved, while runtime-only overlay choices still reset on launch.

The collapsed summary reports Overlay state, visible box types, and HUD state
or corner without exposing stale detection runtime data.

### 05 - Settings

`Settings` contains:

- sound-feedback enabled state;
- Volume slider and exact entry;
- armed-cue preview;
- disabled-cue preview;
- the existing light/dark Theme action.

The oversized volume readout is removed. The exact entry beside the slider is
the authoritative compact readout. Theme appears in this section only and is
not duplicated in the top bar.

The collapsed summary reports sound state, Volume, and current theme.

## Visual hierarchy and styling

- Reuse the existing palette, font family, shared styles, slider implementation,
  state colors, and theme behavior.
- Replace 22-point page titles, eyebrow labels, repeated subtitles, and large
  card descriptions with compact section headers and optional short labels.
- Keep category names visually stronger than field labels, but avoid hero-scale
  typography anywhere inside the fixed window.
- Use cyan with dark readable text for selected/primary actions.
- Keep secondary controls readable with light text in every state.
- Keep the red `STOP` button visually dominant and unchanged in meaning.
- Use labeled buttons for Reconnect, Test, sound previews, and model actions;
  do not depend on icon-only recognition for these controls.
- Preserve focus indicators and normal Tk keyboard traversal.

## Presentation architecture

The implementation remains at the Tk presentation boundary.

### `jitter_app/presentation/widgets.py`

Add the reusable `CollapsibleSection` widget. It manages only:

- header composition;
- expanded/collapsed geometry;
- accessible toggle bindings;
- summary text presentation;
- palette/style refresh hooks needed by theme changes.

It must not read configuration, call services, or own application behavior.
The existing `LiquidNavigation` class may remain available even though the new
application shell no longer instantiates it.

### `jitter_app/presentation/ui.py`

Restructure `_build_page` into:

1. fixed top bar;
2. one dashboard scroll canvas and content frame;
3. five collapsible sections;
4. fixed footer;
5. fixed runtime dock.

Existing builders should be reduced or split into section-content builders and
should reparent existing controls instead of recreating their behavior. All
runtime-observed widget attributes, Tk variables, command callbacks, and the
explicitly preserved `control_frame` and `quick_frame` seams remain available.
Navigation-only containers and layout assertions may be removed or replaced by
section equivalents.

A small main-thread summary refresh routine may combine already validated
values into the five header summaries. Existing setting-change and runtime-state
refresh paths invoke it after their normal work. Summary failure must never
interrupt movement or service state; safe fallback text is sufficient.

### Unchanged modules

No behavior change is required in:

- `jitter_app/motion/`;
- `jitter_app/ai/`;
- `jitter_app/device/`;
- `jitter_app/config/`;
- `jitter_app/presentation/overlay.py`;
- `jitter_app/presentation/sound.py`.

No configuration schema change or migration is introduced.

## Threading and lifecycle

- All section, summary, widget, and Tk-variable access stays on the Tk thread.
- Existing device callbacks and workers continue to marshal UI updates through
  the established queue/`after` paths.
- Expanding or collapsing a section has no inference or movement side effect.
- Overlay visibility remains the only independent overlay inference demand.
- STOP, disconnect, source changes, hotkey disable, model switching, test
  cancellation, AI failure recovery, and shutdown retain their current
  immediate-cancellation behavior.

## Error handling

- Concise validation and operational errors remain in the fixed footer.
- Detailed diagnostics remain in `app.log` through existing logging.
- Invalid numeric entries retain the current error styling and validated-value
  behavior even when their section is subsequently collapsed.
- An AI runtime error still hides the overlay and deselects AI Aim; the layout
  must not intercept or reinterpret that state transition.

## Test-driven implementation strategy

Tests are written or updated before production layout changes.

1. Add failing presentation tests for the fixed single-page shell, absence of
   visible navigation, five ordered sections, and the startup expansion state.
2. Add failing widget tests for independent multi-section expansion and
   keyboard toggling.
3. Add failing layout tests showing that the footer and runtime dock remain
   outside the scroll canvas and `STOP` stays mapped while all sections are
   expanded and content is scrolled.
4. Add failing ownership tests proving every existing control is present in its
   approved section and each numeric control still has a slider and exact entry.
5. Add failing summary tests for current values and safe no-source/disconnected
   states.
6. Implement the minimum presentation changes needed to pass those tests.
7. Preserve and run existing behavioral tests for source selection, Master,
   STOP, Test, model selection, overlay, sound, configuration, and shutdown.

Tests tied only to the old rail, old page titles, or old card geometry should be
replaced with assertions for the approved single-page design. Behavioral tests
must not be weakened to accommodate the layout change.

## Verification

After implementation, run the repository-mandated checks:

```powershell
$jitterSources = @('main.py', 'distribution_metadata.py') + @(Get-ChildItem -LiteralPath 'jitter_app' -Recurse -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
python -m py_compile @jitterSources
python -m unittest discover -s tests -v
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
```

Do not run Nuitka for this ordinary presentation change.

## Acceptance criteria

- The application remains fixed at `840x620` and opens as one English page.
- There is no left navigation rail or visible page-navigation control.
- The five approved sections appear in the approved order.
- Only `Control` is expanded at startup.
- Multiple sections can remain expanded simultaneously.
- All existing controls are available in their approved category.
- Every numeric setting retains both a slider and exact-value entry.
- The footer and `Master / Runtime / STOP` dock are always visible.
- The content uses one scrollbar and no nested scroll regions.
- AI runtime metrics appear only in the overlay HUD, not in the main UI.
- Theme control appears only in `Settings`.
- No section expansion state or new runtime state is persisted.
- Motion, AI, overlay, Makcu, sound, configuration, cancellation, and shutdown
  behavior remain unchanged.
- The full required verification suite passes.
