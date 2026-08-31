# Dual AI Capture Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the original centered 320-by-320 DXCam path as the startup default and add a runtime-selectable full-primary-display path with correct, generation-safe AI and Overlay coordinates.

**Architecture:** One AI generation owns one detector and one mode-specific DXCam capture. The capture boundary returns an owned RGB frame with its primary-output viewport; targeting stays relative to that captured frame while Overlay projection translates the viewport into its full-screen canvas. A runtime-only Tk combobox replaces the AI generation on a live mode change while preserving successful Master/source/Jitter/Overlay state.

**Tech Stack:** Python 3.12, Tkinter/ttk, NumPy, DXCam, ONNX Runtime DirectML, `unittest`, existing Makcu integration.

**Spec:** `docs/superpowers/specs/2026-08-31-dual-ai-capture-mode-design.md`

## Global Constraints

- Windows-only; ONNX Runtime DirectML, DXCam, and NumPy remain the complete approved AI stack.
- Add no dependency, model, download, profile, tray behavior, or configuration-schema field.
- `Center 320` is the default on every process launch; capture mode is runtime-only and never serialized.
- Use exactly one capture, detector, and AI worker at a time; never infer center and full display concurrently.
- Preserve one aspect-preserving letterboxed base inference and at most one existing same-frame Adaptive Zoom refinement.
- Preserve the fixed 1,000 Hz absolute-deadline motion servo, zero-delta suppression, freshness, clamping, and immediate cancellation behavior.
- Keep Tk widget/variable access on the main thread and reject stale work through existing UI epochs and service generations.
- Preserve model validation/rollback, Trigger/Modifier, Master, source, STOP, Test 3s, reconnect, Overlay, and shutdown behavior except for the approved mode-switch lifecycle.
- Keep one primary-display-sized, click-through, capture-excluded Overlay window in both modes.
- Do not touch, stage, copy, package, or persist any runtime-browsed external `.onnx` file.
- Do not run Nuitka.
- Every task must finish with its focused tests and the complete suite green before commit.

## File Structure

- `jitter_app/ai/targeting.py`: immutable detection viewport fields and validation.
- `jitter_app/ai/zoom.py`: refinement preserves its base viewport.
- `jitter_app/presentation/overlay.py`: capture-origin-to-full-canvas projection.
- `jitter_app/ai/capture.py`: mode constants, centered/full physical regions, and atomic captured-frame boundary.
- `jitter_app/ai/service.py`: immutable mode per generation and atomic captured-frame publication.
- `jitter_app/presentation/ui.py`: compact selector and live AI-generation replacement.
- `tests/test_ai_targeting.py`, `tests/test_ai_zoom.py`, `tests/test_overlay.py`: pure viewport behavior.
- `tests/test_ai_capture.py`, `tests/test_ai_service.py`: hardware-free capture/service boundaries and generation safety.
- `tests/test_ui.py`: layout, runtime-only state, switching, guards, failures, and preservation.
- `tests/test_entrypoints.py`, `README.md`, `AGENTS.md`: current dual-mode contract.
- `docs/superpowers/specs/2026-08-31-full-primary-display-detection-design.md`: narrow historical supersession note.

---

### Task 1: Make Published Detections Viewport-Aware

**Files:**
- Modify: `jitter_app/ai/targeting.py:28-55,257-322`
- Modify: `jitter_app/ai/zoom.py:264-344`
- Modify: `jitter_app/ai/service.py:95-120,430-455`
- Modify: `jitter_app/presentation/overlay.py:103-159`
- Modify: `tests/test_ai_targeting.py:180-240`
- Modify: `tests/test_ai_zoom.py:350-560`
- Modify: `tests/test_ai_service.py:1600-1690`
- Modify: `tests/test_overlay.py:15-205`

**Interfaces:**
- Produces appended `DetectionFrameSnapshot` fields: `output_width: int | None = None`, `output_height: int | None = None`, `capture_left: int = 0`, `capture_top: int = 0`.
- Produces `analyze_detections(..., output_width: int | None = None, output_height: int | None = None, capture_left: int = 0, capture_top: int = 0)`.
- Produces viewport-preserving zoom/service snapshot reconstruction.
- Produces Overlay mapping from captured-frame coordinates into output/canvas coordinates.
- Consumes no later-task interface.

- [ ] **Step 1: Add failing targeting viewport tests**

Append fields only after the existing six dataclass fields so all current
positional constructors remain valid. Add:

```python
def test_analysis_publishes_capture_viewport_without_changing_target_center(self):
    result = analyze_detections(
        (Detection(150, 150, 170, 170, 0.9, 7),),
        AimSettings(),
        sequence=4,
        captured_at=2.0,
        frame_width=320,
        frame_height=320,
        output_width=1920,
        output_height=1080,
        capture_left=800,
        capture_top=380,
    )
    self.assertEqual((result.target.aim_x, result.target.aim_y), (160.0, 160.0))
    self.assertEqual(
        (
            result.frame.output_width,
            result.frame.output_height,
            result.frame.capture_left,
            result.frame.capture_top,
        ),
        (1920, 1080, 800, 380),
    )

def test_omitted_viewport_retains_legacy_frame_relative_contract(self):
    result = analyze_detections(
        (), AimSettings(), sequence=1, captured_at=1.0,
        frame_width=640, frame_height=360,
    )
    self.assertIsNone(result.frame.output_width)
    self.assertIsNone(result.frame.output_height)
    self.assertEqual((result.frame.capture_left, result.frame.capture_top),
                     (0, 0))
```

Add subtests for one missing output dimension, booleans/floats, non-positive
outputs, negative origins, and `capture_left + frame_width > output_width` or
`capture_top + frame_height > output_height`. Each explicit invalid viewport
must raise `ValueError("capture viewport must fit the primary output")`.

- [ ] **Step 2: Run targeting tests and verify RED**

```powershell
python -m unittest tests.test_ai_targeting -v
```

Expected: unexpected viewport keywords and missing snapshot fields.

- [ ] **Step 3: Append fields and implement pure viewport validation**

Use this exact dataclass tail:

```python
@dataclass(frozen=True)
class DetectionFrameSnapshot:
    sequence: int
    captured_at: float
    detections: tuple[Detection, ...]
    selected_index: int | None
    frame_width: int = 320
    frame_height: int = 320
    output_width: int | None = None
    output_height: int | None = None
    capture_left: int = 0
    capture_top: int = 0
```

In `analyze_detections`, allow the exact legacy combination
`output_width is output_height is None` only when both origins are zero. For an
explicit viewport, require strict integer positive output dimensions, strict
integer nonnegative origins, and containment of the whole captured frame.
Pass the four values into `DetectionFrameSnapshot`. Do not add output geometry
to `TargetSnapshot`; movement remains relative to `(frame_width/2,
frame_height/2)`.

- [ ] **Step 4: Run targeting tests and verify GREEN**

```powershell
python -m unittest tests.test_ai_targeting -v
```

Expected: all targeting tests pass, including unchanged equality fixtures.

- [ ] **Step 5: Add failing viewport preservation tests**

In `tests/test_ai_zoom.py`, construct a base frame with
`output_width=1920`, `output_height=1080`, `capture_left=800`, and
`capture_top=380`; assert `compose_zoom_refinement()` retains all four values.

In `tests/test_ai_service.py`, place an explicit-viewport snapshot into
`service._latest_detection`, call `reset_targeting()`, and assert the rebuilt
snapshot changes only `selected_index` to `None`. Add an equivalent assertion
to the existing targeting-revision mismatch publication test.

- [ ] **Step 6: Run preservation tests and verify RED**

```powershell
python -m unittest tests.test_ai_zoom tests.test_ai_service -v
```

Expected: rebuilt frames revert to omitted/zero viewport values.

- [ ] **Step 7: Preserve the viewport at every same-frame reconstruction**

Append these fields from the original frame at each reconstruction site:

```python
frame.output_width,
frame.output_height,
frame.capture_left,
frame.capture_top,
```

Apply this to `compose_zoom_refinement()`, `AiService.reset_targeting()`, and
the targeting-revision mismatch publication in `_worker()`. Do not copy these
fields between different captured frames.

- [ ] **Step 8: Add failing centered and full-display Overlay tests**

Add exact center placement:

```python
def test_center_320_projection_uses_physical_capture_origin(self):
    frame = DetectionFrameSnapshot(
        1, 10.0,
        (Detection(0, 0, 320, 320, 0.9, 0),),
        0, 320, 320, 1920, 1080, 800, 380,
    )
    boxes = project_overlay_boxes(
        frame, 10.0, canvas_width=1920, canvas_height=1080
    )
    self.assertEqual(
        (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2),
        (800.0, 380.0, 1120.0, 700.0),
    )

def test_center_320_projection_scales_output_space_to_canvas(self):
    frame = DetectionFrameSnapshot(
        1, 10.0,
        (Detection(0, 0, 320, 320, 0.9, 0),),
        0, 320, 320, 1920, 1080, 800, 380,
    )
    boxes = project_overlay_boxes(
        frame, 10.0, canvas_width=960, canvas_height=540
    )
    self.assertEqual(
        (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2),
        (400.0, 190.0, 560.0, 350.0),
    )
```

Retain the existing full-display scaling test. Add portrait output, invalid
viewport returning `()`, and source clipping before origin translation. Keep
selected-index emphasis tied to the original detection tuple index.

- [ ] **Step 9: Run Overlay tests and verify RED**

```powershell
python -m unittest tests.test_overlay -v
```

Expected: Center 320 boxes are stretched across the full canvas.

- [ ] **Step 10: Implement viewport-aware Overlay projection**

Resolve omitted output dimensions to the source frame dimensions and require
zero origins in that legacy case. Validate explicit geometry with the same
containment contract as targeting. Clip each detection to source-frame bounds,
then calculate:

```python
x_scale = canvas_width / output_width
y_scale = canvas_height / output_height
x1 = (capture_left + source_x1) * x_scale
y1 = (capture_top + source_y1) * y_scale
x2 = (capture_left + source_x2) * x_scale
y2 = (capture_top + source_y2) * y_scale
```

Preserve freshness, head/player filters, labels, selected width, HUD placement,
and final canvas clipping.

- [ ] **Step 11: Run focused and complete tests**

```powershell
python -m unittest tests.test_ai_targeting tests.test_ai_zoom tests.test_ai_service tests.test_overlay -v
python -m unittest discover -s tests -v
```

Expected: both commands pass.

- [ ] **Step 12: Commit Task 1**

```powershell
git add -- jitter_app/ai/targeting.py jitter_app/ai/zoom.py jitter_app/ai/service.py jitter_app/presentation/overlay.py tests/test_ai_targeting.py tests/test_ai_zoom.py tests/test_ai_service.py tests/test_overlay.py
git diff --cached --check
git commit -m "feat: project ai capture viewports"
```

---

### Task 2: Add the Dual Physical Capture and Mode-Per-Generation Service

**Files:**
- Modify: `jitter_app/ai/capture.py:1-72`
- Modify: `jitter_app/ai/service.py:1-490`
- Modify: `tests/test_ai_capture.py:1-145`
- Modify: `tests/test_ai_service.py:1-2320`

**Interfaces:**
- Produces `CENTER_320 = "center_320"`, `FULL_DISPLAY = "full_display"`, `CAPTURE_MODES`, and `validated_capture_mode(raw: object) -> str`.
- Produces `centered_region(width: int, height: int, size: int = 320)` and retains `full_output_region(...)`.
- Produces frozen `CapturedFrame(pixels, output_width, output_height, capture_left, capture_top, capture_width, capture_height, mode)`.
- Produces `DxcamCapture(..., mode: str = CENTER_320, target_fps: int = 120)` with `read() -> CapturedFrame | None`.
- Produces `AiService.start(..., model_path=None, capture_mode: str = CENTER_320) -> int | None`.
- Produces constructor seam `capture_factory: Callable[[str], Any] | None`, invoked once with the immutable generation mode.
- Consumes Task 1 explicit viewport publication.

- [ ] **Step 1: Add failing strict mode and region tests**

Import the planned symbols and add:

```python
def test_capture_modes_are_strict(self):
    self.assertEqual(validated_capture_mode(CENTER_320), CENTER_320)
    self.assertEqual(validated_capture_mode(FULL_DISPLAY), FULL_DISPLAY)
    for invalid in (None, "", "center", "CENTER_320", 320, True):
        with self.subTest(invalid=invalid):
            with self.assertRaisesRegex(
                ValueError, "^Unsupported AI capture mode$"
            ):
                validated_capture_mode(invalid)

def test_centered_region_restores_exact_320_geometry(self):
    self.assertEqual(centered_region(1920, 1080), (800, 380, 1120, 700))
    self.assertEqual(centered_region(1919, 1079), (799, 379, 1119, 699))
    self.assertEqual(centered_region(320, 320), (0, 0, 320, 320))
    with self.assertRaisesRegex(ValueError, "smaller than"):
        centered_region(319, 1080)
```

Add strict invalid width/height/size cases: zero, negative, boolean, float, and
string values.

- [ ] **Step 2: Run region tests and verify RED**

```powershell
python -m unittest tests.test_ai_capture.CaptureTests.test_capture_modes_are_strict tests.test_ai_capture.CaptureTests.test_centered_region_restores_exact_320_geometry -v
```

Expected: import failures for missing constants/helper/validator.

- [ ] **Step 3: Add failing physical capture tests**

Center must be the constructor default:

```python
def test_default_capture_requests_center_320_and_returns_atomic_geometry(self):
    source = np.zeros((320, 320, 3), dtype=np.uint8)
    camera = FakeCamera(width=1920, height=1080, frame=source)
    capture = DxcamCapture(
        camera_factory=RecordingCameraFactory(camera), target_fps=165
    )
    capture.start()
    captured = capture.read()

    self.assertIsInstance(captured, CapturedFrame)
    self.assertEqual(camera.start_kwargs, {
        "region": (800, 380, 1120, 700), "target_fps": 165,
    })
    self.assertEqual(
        (captured.output_width, captured.output_height,
         captured.capture_left, captured.capture_top,
         captured.capture_width, captured.capture_height, captured.mode),
        (1920, 1080, 800, 380, 320, 320, CENTER_320),
    )
    self.assertTrue(captured.pixels.flags.owndata)
    self.assertTrue(captured.pixels.flags.c_contiguous)
    self.assertFalse(np.shares_memory(captured.pixels, source))

def test_full_display_capture_requests_native_output(self):
    source = np.zeros((1080, 1920, 3), dtype=np.uint8)
    camera = FakeCamera(width=1920, height=1080, frame=source)
    capture = DxcamCapture(
        camera_factory=RecordingCameraFactory(camera), mode=FULL_DISPLAY
    )
    capture.start()
    captured = capture.read()
    self.assertEqual(camera.start_kwargs["region"], (0, 0, 1920, 1080))
    self.assertEqual(captured.pixels.shape, (1080, 1920, 3))
    self.assertEqual(captured.mode, FULL_DISPLAY)
```

Add exact shape-agreement failure: valid RGB `(320,320,3)` returned for a
full `(1920,1080)` capture raises
`ValueError("AI capture frame must match capture region")`. Preserve `None`,
malformed format, idempotent close, and release-after-stop-error coverage.

- [ ] **Step 4: Run capture module and verify RED**

```powershell
python -m unittest tests.test_ai_capture -v
```

Expected: full-only region and bare-array return failures.

- [ ] **Step 5: Implement the capture constants, frame type, and mode-specific DXCam region**

Use exact public definitions:

```python
CENTER_320 = "center_320"
FULL_DISPLAY = "full_display"
CAPTURE_MODES = (CENTER_320, FULL_DISPLAY)

def validated_capture_mode(raw: Any) -> str:
    if type(raw) is not str or raw not in CAPTURE_MODES:
        raise ValueError("Unsupported AI capture mode")
    return raw

@dataclass(frozen=True)
class CapturedFrame:
    pixels: np.ndarray
    output_width: int
    output_height: int
    capture_left: int
    capture_top: int
    capture_width: int
    capture_height: int
    mode: str
```

`centered_region()` uses strict positive integers, rejects a smaller output,
and floors left/top. `DxcamCapture.__init__` validates/stores mode. `start()`
strictly validates camera dimensions, selects the region, starts DXCam, and
stores active geometry only after successful start. `read()` retains RGB
`uint8` validation, requires exact active-region shape, creates one owned
C-contiguous copy, and returns `CapturedFrame`. `close()` clears active
geometry and preserves current stop/release safety.

- [ ] **Step 6: Run capture module and verify GREEN**

```powershell
python -m unittest tests.test_ai_capture -v
```

Expected: all capture tests pass.

- [ ] **Step 7: Migrate the service test seam to explicit captured frames**

Keep `rgb_frame()` for detector/zoom arrays and add:

```python
def captured_frame(
    pixels=None, *, output_width=None, output_height=None,
    capture_left=0, capture_top=0, mode=CENTER_320,
):
    pixels = rgb_frame() if pixels is None else pixels
    height, width = pixels.shape[:2]
    return CapturedFrame(
        np.array(pixels, copy=True, order="C"),
        width if output_width is None else output_width,
        height if output_height is None else output_height,
        capture_left, capture_top, width, height, mode,
    )
```

Make test-only `FakeCapture` wrap bare NumPy fixtures with this helper while
preserving already-wrapped values. Change all service `capture_factory`
callables to one mode argument, for example `lambda _mode: capture`. Change
`FalseyCaptureFactory.__call__(self, mode)` similarly. This compatibility is
test-only; production service consumes `CapturedFrame` exclusively.

Any existing service test that intentionally uses a non-320 native frame must
wrap it with `mode=FULL_DISPLAY` and start the service with
`capture_mode=FULL_DISPLAY`. Do not label a 640-by-360 or other rectangular
fixture as Center 320 merely to keep an old test green.

- [ ] **Step 8: Add failing service mode and geometry tests**

```python
def test_start_copies_full_display_mode_into_generation_capture(self):
    modes = []
    captured = captured_frame(
        rgb_frame(width=1920, height=1080),
        output_width=1920, output_height=1080,
        mode=FULL_DISPLAY,
    )
    service = AiService(
        lambda _event: None,
        detector_factory=lambda _path: SequenceDetector([()]),
        capture_factory=lambda mode: modes.append(mode) or FakeCapture([captured]),
    )
    self.addCleanup(service.close)
    service.start(AimSettings, capture_mode=FULL_DISPLAY)
    self.assertTrue(wait_until(
        lambda: service.latest_detection_snapshot() is not None
    ))
    self.assertEqual(modes, [FULL_DISPLAY])
    frame = service.latest_detection_snapshot()
    self.assertEqual(
        (frame.frame_width, frame.frame_height,
         frame.output_width, frame.output_height,
         frame.capture_left, frame.capture_top),
        (1920, 1080, 1920, 1080, 0, 0),
    )

def test_invalid_capture_mode_fails_before_service_state_changes(self):
    events = []
    service = AiService(events.append, capture_factory=lambda _mode: None)
    self.addCleanup(service.close)
    with self.assertRaisesRegex(ValueError, "^Unsupported AI capture mode$"):
        service.start(AimSettings, capture_mode="wide")
    self.assertFalse(service.running)
    self.assertEqual(events, [])
```

Add a centered captured frame with output `(1920,1080)`, origin `(800,380)`,
and frame `(320,320)`. Assert targeting still centers on `(160,160)` and the
detection snapshot publishes the explicit viewport.

- [ ] **Step 9: Run new service tests and verify RED**

```powershell
python -m unittest tests.test_ai_service.AiServiceTests.test_start_copies_full_display_mode_into_generation_capture tests.test_ai_service.AiServiceTests.test_invalid_capture_mode_fails_before_service_state_changes -v
```

Expected: missing `capture_mode`, mode-unaware factory, and bare-array worker.

- [ ] **Step 10: Implement mode-per-generation service consumption**

Import the new interfaces from `jitter_app.ai.capture`. Store a one-mode
capture factory:

```python
self._capture_factory = (
    capture_factory
    if capture_factory is not None
    else lambda mode: DxcamCapture(mode=mode, target_fps=capture_fps)
)
```

Validate `capture_mode` at the first line of `start()` before state mutation,
copy it into `_worker` arguments, and call the factory exactly once with that
generation value. In the loop:

```python
captured = capture.read()
if captured is None:
    stop_event.wait(0.001)
    continue
frame = captured.pixels
frame_height, frame_width = frame.shape[:2]
```

Require captured mode, explicit region size, pixel shape, positive output, and
contained origin to match. Additionally require Center 320 geometry to be an
exact centered 320-by-320 region and Full Display geometry to have zero origin
with capture dimensions equal to the output. Otherwise raise
`ValueError("AI captured frame geometry is inconsistent")`. Pass output
dimensions and origin into `analyze_detections()`. Detector and Adaptive Zoom
receive only `frame`.

Update default-factory tests to assert both `mode` and `target_fps`. Preserve
the falsey injected factory's authority.

- [ ] **Step 11: Add stale-old-viewport restart coverage**

Start a blocked Center 320 generation, stop it, then start a publishing Full
Display generation. Release the first detector only after the second snapshot
is visible. Assert the latest snapshot remains Full Display and no late center
ready/error/detection replaces it. Reuse existing blocking detector, service
generation, and event-barrier helpers rather than adding sleeps.

- [ ] **Step 12: Run focused and complete tests**

```powershell
python -m unittest tests.test_ai_capture tests.test_ai_service tests.test_ai_targeting tests.test_ai_zoom tests.test_overlay -v
python -m unittest discover -s tests -v
```

Expected: both commands pass.

- [ ] **Step 13: Commit Task 2**

```powershell
git add -- jitter_app/ai/capture.py jitter_app/ai/service.py tests/test_ai_capture.py tests/test_ai_service.py
git diff --cached --check
git commit -m "feat: bind ai generations to capture modes"
```

---

### Task 3: Add the Compact Runtime Selector and Live-Switch Lifecycle

**Files:**
- Modify: `jitter_app/presentation/ui.py:1-1205,1585-1650,2689-2750,2915-3205,3440-3680,4040-4250`
- Modify: `tests/test_ui.py:220-315,440-570,590-660,1080-1170,1520-1580,2585-2665,2800-3500,4000-4090`

**Interfaces:**
- Consumes Task 2 `CENTER_320`, `FULL_DISPLAY`, `validated_capture_mode`, and `AiService.start(..., capture_mode=...)`.
- Produces `_capture_mode: str`, `_capture_mode_switching: bool`, `capture_mode_var`, `capture_mode_combo`, `_capture_mode_changed()`, and `_render_capture_mode_control()`.
- Uses labels `Center 320` and `Full Display` mapped exactly to Task 2 constants.

- [ ] **Step 1: Extend UI AI-service stubs**

Change both `StubAiService.start()` implementations to accept
`capture_mode=CENTER_320` and record:

```python
(settings_provider, zoom_gate_provider, model_path, capture_mode)
```

Keep model path at index 2. Update the single three-name tuple unpack to four
names and assert existing initial starts carry `CENTER_320`.

- [ ] **Step 2: Add failing startup, layout, palette, summary, and persistence tests**

```python
def test_capture_mode_starts_centered_and_shares_target_row(self):
    self.assertEqual(self.app._capture_mode, CENTER_320)
    self.assertEqual(self.app.capture_mode_var.get(), "Center 320")
    self.assertEqual(
        tuple(self.app.capture_mode_combo.cget("values")),
        ("Center 320", "Full Display"),
    )
    self.assertEqual(
        self.app.capture_mode_combo.master.master,
        self.app.target_area_combo.master.master,
    )

def test_capture_mode_is_runtime_only_and_new_app_restores_center(self):
    self.app.capture_mode_var.set("Full Display")
    self.app._capture_mode_changed()
    self.app.save_config()
    self.assertFalse(hasattr(self.store.saved[-1], "capture_mode"))
    config = self.app.config
    self.app.close_app()
    app = self.make_app(config=config)
    self.assertEqual(app._capture_mode, CENTER_320)
```

Add `capture_mode_combo` to the popup-palette test. Assert the collapsed AI
summary contains `Center 320` and retains its existing global length bound.
Assert the fixed outer geometry and five-section order do not change.

- [ ] **Step 3: Run layout tests and verify RED**

```powershell
python -m unittest tests.test_ui.JitterLayoutTests.test_capture_mode_starts_centered_and_shares_target_row tests.test_ui.JitterLayoutTests.test_capture_mode_is_runtime_only_and_new_app_restores_center tests.test_ui.JitterLayoutTests.test_combobox_popups_use_liquid_colors_in_both_themes -v
```

Expected: missing runtime state/widgets.

- [ ] **Step 4: Build the compact runtime-only control**

Initialize `_capture_mode = CENTER_320` and
`_capture_mode_switching = False` with AI lifecycle state. Create
`capture_mode_var = tk.StringVar(self, "Center 320")` without reading config.

Replace the full-width Target Area row with one two-column surface frame using
equal `uniform` weights. Place Target Area in column 0 and Capture Mode in
column 1 through `_dropdown_field`, then bind `<<ComboboxSelected>>`. Add the
combo to `_apply_combobox_popup_palette()`.

Include the capture label in `_refresh_section_summaries()` and reduce only the
model-label allowance so `_compact_section_summary` keeps its current bound.
Implement `_render_capture_mode_control()` using `readonly` normally and
`disabled` during shutdown, Test 3s, model transitions, capture restart, and
initial non-ready AI loading. Call it from both runtime/model render paths.

- [ ] **Step 5: Run layout tests and verify GREEN**

Run Step 3 plus:

```powershell
python -m unittest tests.test_ui.JitterLayoutTests.test_dashboard_has_five_ordered_independent_sections tests.test_ui.JitterLayoutTests.test_window_is_fixed_size_liquid_split_console tests.test_ui.JitterLayoutTests.test_each_existing_control_belongs_to_its_approved_section -v
```

Expected: all pass with 840-by-620 outer geometry unchanged.

- [ ] **Step 6: Add failing idle and active switch tests**

```python
def test_idle_capture_mode_change_does_not_start_ai(self):
    self.app.capture_mode_var.set("Full Display")
    self.app._capture_mode_changed()
    self.assertEqual(self.app._capture_mode, FULL_DISPLAY)
    self.assertEqual(self.ai.start_calls, [])
    self.assertEqual(self.ai.stop_calls, [])

def test_active_capture_mode_change_restarts_ai_and_keeps_overlay(self):
    self.app.toggle_overlay()
    self.ai.emit(AiEvent("ready", "DmlExecutionProvider"))
    starts = len(self.ai.start_calls)
    self.app.capture_mode_var.set("Full Display")
    self.app._capture_mode_changed()
    self.assertTrue(self.app.overlay_visible)
    self.assertEqual(self.ai.stop_calls[-1], "Capture mode changed")
    self.assertEqual(len(self.ai.start_calls), starts + 1)
    self.assertEqual(self.ai.start_calls[-1][3], FULL_DISPLAY)
    self.assertEqual(self.app.capture_mode_combo.cget("state"), "disabled")
    self.ai.emit(AiEvent("ready", "DmlExecutionProvider"))
    self.assertEqual(self.app.capture_mode_combo.cget("state"), "readonly")
```

Add a connected, Master-armed, Trigger-held, combined Jitter+AI case. Capture
the Makcu motion generation and cancellation count before switching. Assert
Master, both source selections, held gate, Overlay visibility, motion
generation, and cancellation count remain unchanged.

- [ ] **Step 7: Run switch tests and verify RED**

```powershell
python -m unittest tests.test_ui.JitterLayoutTests.test_idle_capture_mode_change_does_not_start_ai tests.test_ui.JitterLayoutTests.test_active_capture_mode_change_restarts_ai_and_keeps_overlay tests.test_ui.JitterLayoutTests.test_combined_motion_continues_during_capture_mode_restart -v
```

Expected: missing handler and missing mode forwarding.

- [ ] **Step 8: Implement guarded generation replacement**

`_capture_mode_changed()` must map/validate the selected label and restore the
current label for invalid or guarded events. Guard shutdown, Test 3s, model
switch, existing capture switch, and initial non-ready AI loading. Same-value
selection is a no-op.

For an accepted idle value, store it, refresh controls/summary, and do not start
AI. For an active AI runtime:

1. Store the new mode and set `_capture_mode_switching = True`.
2. Call `ai_service.reset_targeting()` and store its revision.
3. Call `_stop_ai_runtime("Capture mode changed")` while the old runtime is
   still marked active, then set `_ai_runtime_active = False`, clear
   ready/provider/FPS, reset zoom, and synchronize the Adaptive Zoom gate.
4. Start one new generation with the same committed model and selected mode.
5. Do not stop the Makcu motion worker, clear selected sources/Master/held
   buttons, or hide Overlay.
6. `_start_ai_runtime()` always passes
   `capture_mode=self._capture_mode`, including normal, Test, model candidate,
   and rollback starts.
7. Current `ready` clears the switch flag, re-renders the combo, and reports
   `AI capture ready: <label>`. Current start/error failure clears the flag
   before entering existing failure policy. Old queued events remain rejected
   by `_ai_event_epoch`.

For a synchronous false/exception start, call
`_hide_overlay_after_ai_failure()` before `_handle_ai_start_failure()` so the
existing fail-closed policy is complete. Do not roll `_capture_mode` back; the
user may explicitly select Center 320 and re-arm after a failed Full Display
start.

- [ ] **Step 9: Add lifecycle guard and failure coverage**

Test all exact cases:

- model validation and every `_TEST_MOTION_MODES` value disable the combo and a
  programmatic event restores the current label without lifecycle calls;
- Test 3s and model candidate/rollback generations carry the mode selected
  before the protected transition;
- STOP keeps `FULL_DISPLAY`, while a newly built app starts at center;
- false and exceptional restart results use existing failure behavior, hide
  Overlay, deselect AI, retain eligible Jitter, retain the requested capture
  mode, and clear switch state;
- STOP, disconnect, or shutdown during a capture restart clears the switching
  flag and cannot leave the combobox lifecycle stuck;
- stale ready/error from the stopped AI generation cannot re-enable or mutate
  the replacement;
- Center-to-Full and Full-to-Center each make exactly one stop and one start;
- AI-only motion remains alive but emits no AI delta while the service snapshot
  is cleared; combined motion continues Jitter without a cancel/restart.

- [ ] **Step 10: Run focused and complete tests**

```powershell
python -m unittest tests.test_ui -v
python -m unittest tests.test_ai_service tests.test_overlay tests.test_ui -v
python -m unittest discover -s tests -v
```

Expected: all three commands pass.

- [ ] **Step 11: Commit Task 3**

```powershell
git add -- jitter_app/presentation/ui.py tests/test_ui.py
git diff --cached --check
git commit -m "feat: switch ai capture modes at runtime"
```

---

### Task 4: Update the Current Contract and Run Canonical Verification

**Files:**
- Modify: `AGENTS.md:20-35,112-205,238-248,285-295`
- Modify: `README.md:45-115,240-270,425-460`
- Modify: `tests/test_entrypoints.py:580-615`
- Modify: `docs/superpowers/specs/2026-08-31-full-primary-display-detection-design.md:1-15,288-300`

**Interfaces:**
- Consumes Tasks 1-3 behavior and exact labels.
- Produces documentation that no longer calls Full Display the sole physical capture path.

- [ ] **Step 1: Add failing bounded README assertions**

Rename the current README test to
`test_readme_documents_supported_sizes_and_both_capture_modes`. Keep its
feature/targeting section slicing and require:

```python
for contract in (
    "Capture Mode",
    "Center 320",
    "Full Display",
    "runtime-only",
    "320×320",
    "unused letterbox pixels are filled with RGB value 114",
):
    with self.subTest(contract=contract):
        self.assertIn(contract, features_section)
```

Require the targeting section to state that the crosshair center comes from
the selected source frame. Require the repository-layout section to describe
`capture.py` as owning centered and full-primary regions. Remove only the old
full-only literal; retain model-size, 128/256 rejection, padding ownership,
source-coordinate, and 1,000 Hz assertions.

- [ ] **Step 2: Run the renamed test and verify RED**

```powershell
python -m unittest tests.test_entrypoints.AppEntrypointTests.test_readme_documents_supported_sizes_and_both_capture_modes -v
```

Expected: README lacks the dual-mode contract.

- [ ] **Step 3: Update README and AGENTS precisely**

Document:

- startup-default `Center 320`, physically captured at the centered square;
- runtime-only `Full Display`, captured at native primary resolution and
  letterboxed with RGB 114;
- exactly one mode/generation at a time;
- live switching clears old publications and replaces only the AI generation,
  preserving successful Master/source/Jitter/Overlay state;
- full-screen Overlay translation for center and full viewports;
- Test 3s holds its selected mode and model candidate/rollback uses it;
- STOP/errors retain the runtime selection, but every launch resets center;
- no config schema field, new dependency, bundled model, or packaging change.

Update the `capture.py` layout description and configuration non-persistence
list. Preserve every external ONNX safety restriction and existing error rule.

- [ ] **Step 4: Mark the older sole-mode spec as narrowly superseded**

Add below its title:

```markdown
> Superseded in one respect by
> `2026-08-31-dual-ai-capture-mode-design.md`: Full Display is selectable,
> while Center 320 is restored as the runtime startup default. The geometry,
> letterbox, targeting, zoom, and Overlay rules here remain active for Full
> Display.
```

Replace its sole-mode UI sentence with a pointer to the new design. Do not
rewrite the historical implementation plan.

- [ ] **Step 5: Run focused documentation/package tests**

```powershell
python -m unittest tests.test_entrypoints tests.test_package_layout tests.test_distribution_metadata -v
```

Expected: all pass.

- [ ] **Step 6: Run source compilation**

```powershell
$jitterSources = @('main.py', 'distribution_metadata.py') + @(Get-ChildItem -LiteralPath 'jitter_app' -Recurse -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
python -m py_compile @jitterSources
```

Expected: exit 0.

- [ ] **Step 7: Run the complete hardware-free suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: every test passes.

- [ ] **Step 8: Run imports and DirectML self-check**

```powershell
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
```

Expected: imports exit 0; self-check reports `"status": "ok"`, the approved
bundled-model hash, and `"provider": "DmlExecutionProvider"`.

- [ ] **Step 9: Review packaging metadata without building**

```powershell
python .\distribution_metadata.py --review-json
```

Expected: exit 0 and only `models/all_games_320.onnx` as bundled model data. Do
not run `gen.bat` or Nuitka.

- [ ] **Step 10: Audit diff scope and protected external models**

```powershell
git diff --check
git status --short
Get-FileHash -Algorithm SHA256 -LiteralPath @(
  'models/Apex_20k_pictures_640.onnx',
  'models/all_games.onnx',
  'models/all_games_128.onnx',
  'models/all_games_256.onnx',
  'models/all_games_640.onnx'
)
```

Expected: no whitespace errors; only intended tracked changes plus the same
pre-existing untracked models; model hashes match the implementation baseline.

- [ ] **Step 11: Commit Task 4**

```powershell
git add -- AGENTS.md README.md tests/test_entrypoints.py docs/superpowers/specs/2026-08-31-full-primary-display-detection-design.md
git diff --cached --check
git commit -m "docs: describe selectable ai capture modes"
```

- [ ] **Step 12: Perform hardware acceptance when Makcu is available**

Run `python main.py` and verify:

1. launch defaults to Center 320 and boxes align to the physical center region;
2. Full Display detects/projects across the primary display;
3. switching both directions during Overlay-only and combined motion preserves
   state and never moves from an old target;
4. Test 3s holds its mode and disables the selector;
5. model candidate/rollback uses the selected mode;
6. Trigger, Modifier, reconnect, STOP, hotkey disable, and shutdown remain
   immediate;
7. Overlay stays click-through and absent from DXCam capture in both modes.

Record hardware-only checks as pending when no device is connected; never claim
them from unit tests.
