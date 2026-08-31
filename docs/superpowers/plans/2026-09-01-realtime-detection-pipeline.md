# Realtime Detection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove avoidable capture copies and overlay polling latency while preserving truthful 150 ms freshness and the fixed 1,000 Hz movement servo.

**Architecture:** DXCam attaches its QPC capture timestamp to the immutable `CapturedFrame`; the generation-safe AI worker validates and publishes that timestamp. The Tk overlay polls at the existing display-derived capture cadence and uses a render fingerprint to redraw only for a new frame, a freshness transition, changed HUD runtime data, or changed immutable style.

**Tech Stack:** Python 3.12, NumPy 2.5.2, DXCam 0.3.0, ONNX Runtime DirectML 1.24.4, Tkinter, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-01-realtime-detection-pipeline-design.md`

## Global Constraints

- Keep Tk widget and Tk-variable access on the main thread.
- Exactly one capture mode and one AI generation run at a time.
- Derive capture cadence from the primary display and cap capture at 240 FPS; fall back to 120 FPS when display detection is unavailable or invalid.
- Run one fixed 1,000 Hz motion servo; use absolute deadlines, skip missed slots, and never queue catch-up movement.
- Consume each fresh AI target through time-based servo microsteps and discard any unconsumed target after 150 ms.
- Preserve Center 320, Full Display, Adaptive Zoom, current-frame stateless selection, generation safety, STOP/disconnect behavior, model contracts, and overlay projection.
- Do not add prediction, tracking, extra workers, frame queues, dependencies, persistence, downloads, external-model copying, packaging, or model changes.
- Use test-driven development: add a failing test, verify the expected failure, implement minimally, then run the complete suite.
- Do not modify or copy the untracked external `.onnx` files from the primary checkout; only the tracked bundled `models/all_games_320.onnx` exists in this worktree.

---

### Task 1: Timestamped single-copy capture and cadence-aware overlay publication

**Files:**
- Modify: `jitter_app/ai/capture.py`
- Modify: `jitter_app/ai/service.py`
- Modify: `jitter_app/presentation/ui.py`
- Modify: `tests/test_ai_capture.py`
- Modify: `tests/test_ai_service.py`
- Modify: `tests/test_ui.py`

**Interfaces:**
- Consumes: DXCam 0.3.0 `get_latest_frame(copy=True, with_timestamp=True) -> (np.ndarray, float) | None`; `RuntimeCadence.capture_fps`; `MAX_FRAME_AGE_S == 0.150` from `jitter_app.presentation.overlay`.
- Produces: `CapturedFrame(..., mode: str, captured_at: float | None = None)`; source timestamps propagated to `TargetSnapshot.captured_at` and `DetectionFrameSnapshot.captured_at`; `_overlay_poll_interval_ms(capture_fps: int) -> int`; duplicate-suppressing `_poll_overlay()` behavior.

- [ ] **Step 1: Write capture tests that prove the timestamp request and single-copy ownership contract**

  Update `FakeCamera` in `tests/test_ai_capture.py` so `copy=True` mirrors real
  DXCam by creating one owned C-order array, stores it as `returned_frame`, and
  returns `(returned_frame, timestamp)` when `with_timestamp=True`. Extend
  `test_default_capture_requests_center_320_and_returns_atomic_geometry` with
  these literal assertions:

  ```python
  self.assertEqual(
      camera.get_latest_frame_kwargs,
      {"copy": True, "with_timestamp": True},
  )
  self.assertIs(frame.pixels, camera.returned_frame)
  self.assertEqual(frame.captured_at, 12.25)
  ```

  Add `test_read_defensively_copies_borrowed_or_noncontiguous_frame` using a
  camera double that returns a non-owning strided `(320, 320, 3)` view plus
  timestamp `4.5`; assert the published pixels are owned, C-contiguous, equal
  to the view, and share no memory. Add
  `test_read_rejects_malformed_timestamped_results` for a bare array, wrong
  tuple length, boolean timestamp, negative timestamp, `nan`, and `inf`.

- [ ] **Step 2: Run capture tests and record the expected RED result**

  Run:

  ```powershell
  python -m unittest tests.test_ai_capture -v
  ```

  Expected RED: existing production code requested only `copy=True`, returned
  no `captured_at`, and made a second array copy; malformed timestamped result
  cases were not rejected by the new contract.

- [ ] **Step 3: Implement timestamped one-copy DXCam capture minimally**

  In `jitter_app/ai/capture.py`, import `math`, append the compatible field,
  and validate the DXCam return at the boundary:

  ```python
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
      captured_at: float | None = None
  ```

  `DxcamCapture.read()` must call
  `get_latest_frame(copy=True, with_timestamp=True)`, accept only an exact
  two-item tuple after the `None` case, require a real finite non-negative
  timestamp while rejecting booleans, and keep the returned owned C-contiguous
  array by identity. Use `np.array(frame, copy=True, order="C")` only when
  `frame.flags.owndata` is false or `frame.flags.c_contiguous` is false. Return
  the normalized `float` timestamp as the final `CapturedFrame` field.

- [ ] **Step 4: Run capture tests and record GREEN**

  Run:

  ```powershell
  python -m unittest tests.test_ai_capture -v
  ```

  Expected GREEN: every capture test passes with no warnings or errors.

- [ ] **Step 5: Write service tests for source-time publication and fail-closed future timestamps**

  Extend the `captured_frame()` test helper in `tests/test_ai_service.py` with
  `captured_at=None` and pass it as the final `CapturedFrame` argument. Add
  `test_worker_publishes_source_capture_time_in_both_snapshots`: use a
  `CapturedFrame` timestamped `9.75`, a `MutableClock(10.0)`, one accepted head,
  and assert both `latest_snapshot().captured_at` and
  `latest_detection_snapshot().captured_at` equal exactly `9.75`. Add
  `test_future_capture_timestamp_uses_runtime_error_cleanup`: use timestamp
  `10.01` with `MutableClock(10.0)` and assert detector publication remains
  empty, the capture closes once, service status is `error`, and the emitted
  safe error payload remains `ValueError: AI service failed`.

- [ ] **Step 6: Run the focused service tests and record the expected RED result**

  Run:

  ```powershell
  python -m unittest tests.test_ai_service.AiServiceTests.test_worker_publishes_source_capture_time_in_both_snapshots tests.test_ai_service.AiServiceTests.test_future_capture_timestamp_uses_runtime_error_cleanup -v
  ```

  Expected RED: snapshots use the service observation time and a future source
  timestamp is not yet validated.

- [ ] **Step 7: Publish the validated source capture time**

  In `jitter_app/ai/service.py`, add one pure normalization function. `None`
  returns the already sampled service observation time for legacy injected
  frames. A provided value must be a non-boolean `int` or `float`, finite,
  non-negative, and no greater than the observation time; otherwise raise
  `ValueError("AI captured frame timestamp is inconsistent")`. In the worker,
  keep the same number and location of service clock samples by replacing the
  old post-validation `captured_at = self._clock()` with:

  ```python
  observed_at = self._clock()
  captured_at = _validated_capture_timestamp(
      captured.captured_at,
      observed_at,
  )
  ```

  Pass that exact value through existing `analyze_detections`; do not change
  inference count, zoom gating, generation checks, publication locking, or FPS
  accounting.

- [ ] **Step 8: Run the complete service tests and record GREEN**

  Run:

  ```powershell
  python -m unittest tests.test_ai_service -v
  ```

  Expected GREEN: every service test passes with only the existing intentional
  logged-error assertions and no new warning/error noise.

- [ ] **Step 9: Write overlay scheduling and render-deduplication tests**

  In `tests/test_ui.py`, add behavior tests that patch `app.after` only after
  construction so the delay passed by `_poll_overlay` can be observed:

  - `test_overlay_poll_uses_display_derived_capture_cadence` creates a
    `RuntimeCadence(240, 240, 1000)`, enables the overlay, and asserts the
    scheduled delay is exactly `4` ms.
  - `test_overlay_poll_skips_duplicate_frame_but_renders_new_sequence` supplies
    immutable `DetectionFrameSnapshot` instances; the second poll of the same
    fresh snapshot leaves the render count unchanged, while replacing it with
    sequence `2` increases the count once.
  - `test_overlay_poll_renders_once_when_frame_becomes_stale` starts at
    timestamp `10.0` with `now=10.10`, advances to `10.151`, then `10.20`, and
    asserts render counts `1`, `2`, `2`.
  - `test_overlay_poll_redraws_when_runtime_or_style_changes` keeps the same
    snapshot, changes the FPS runtime string and then `overlay_player_visible`,
    and asserts one redraw for each change.

  Cancel the currently scheduled `_overlay_after_id` before manually invoking
  another poll so each test owns its callbacks.

- [ ] **Step 10: Run the focused UI tests and record the expected RED result**

  Run:

  ```powershell
  python -m unittest tests.test_ui.JitterLayoutTests.test_overlay_poll_uses_display_derived_capture_cadence tests.test_ui.JitterLayoutTests.test_overlay_poll_skips_duplicate_frame_but_renders_new_sequence tests.test_ui.JitterLayoutTests.test_overlay_poll_renders_once_when_frame_becomes_stale tests.test_ui.JitterLayoutTests.test_overlay_poll_redraws_when_runtime_or_style_changes -v
  ```

  Expected RED: the current fixed delay is 16 ms and every poll calls
  `overlay.render()`.

- [ ] **Step 11: Implement cadence-aware overlay polling with a render fingerprint**

  In `jitter_app/presentation/ui.py`, import `MAX_FRAME_AGE_S`, add the pure
  helper below, and store its result during initialization:

  ```python
  def _overlay_poll_interval_ms(capture_fps: int) -> int:
      if type(capture_fps) is not int or capture_fps <= 0:
          capture_fps = 120
      return max(1, int(1000 / min(capture_fps, 240)))
  ```

  Initialize `_overlay_poll_delay_ms` from
  `self.runtime_cadence.capture_fps` and initialize
  `_last_overlay_render_key = None`. Build a hashable render key from:

  ```python
  (
      (snapshot.sequence, snapshot.captured_at, is_fresh)
          if snapshot exposes those fields
          else ("opaque", id(snapshot)),
      runtime,
      style,
  )
  ```

  Use the overlay's exact freshness rule:
  `max(0.0, now - snapshot.captured_at) <= MAX_FRAME_AGE_S`; use a stable
  `("none",)` token for `snapshot is None`. Call `overlay.render()` and update
  the stored key only when the key changes. Always reschedule the lightweight
  poll with `_overlay_poll_delay_ms`. Reset the stored key before/after an
  overlay show and in the centralized hide/error paths so reopening always
  produces an initial draw. Keep all this work on the Tk thread.

- [ ] **Step 12: Run complete focused component suites and record GREEN**

  Run:

  ```powershell
  python -m unittest tests.test_ai_capture tests.test_ai_service tests.test_ui -v
  ```

  Expected GREEN: all capture, service, and UI tests pass.

- [ ] **Step 13: Run the complete repository verification**

  Run exactly:

  ```powershell
  $jitterSources = @('main.py', 'distribution_metadata.py') + @(Get-ChildItem -LiteralPath 'jitter_app' -Recurse -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
  python -m py_compile @jitterSources
  python -m unittest discover -s tests -v
  python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
  python .\main.py --ai-runtime-self-check
  python .\distribution_metadata.py --review-json
  ```

  Expected GREEN: compile/import commands exit zero, all tests pass, the
  runtime self-check reports `DmlExecutionProvider`, and release review JSON
  reports no errors. Do not run Nuitka.

- [ ] **Step 14: Self-review scope and commit the task**

  Inspect `git diff --check`, `git status --short`, and the complete diff.
  Confirm no model, config, log, packaging, dependency, servo, detector,
  targeting, or unrelated UI behavior changed. Then commit only the listed
  plan, spec, production, and test files:

  ```powershell
  git add docs/superpowers/specs/2026-09-01-realtime-detection-pipeline-design.md docs/superpowers/plans/2026-09-01-realtime-detection-pipeline.md jitter_app/ai/capture.py jitter_app/ai/service.py jitter_app/presentation/ui.py tests/test_ai_capture.py tests/test_ai_service.py tests/test_ui.py
  git commit -m "perf: reduce realtime detection latency"
  ```
