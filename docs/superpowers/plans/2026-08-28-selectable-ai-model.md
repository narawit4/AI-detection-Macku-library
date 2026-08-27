# Selectable AI Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user select a compatible external ONNX model for the current Jitter process, switch generations safely, and automatically restore the previous model after candidate failure.

**Architecture:** A new `ai_model_selection.py` module owns immutable runtime model choices, cheap path validation, and a daemon one-shot contract validator that never creates DXCam or runs inference. `AiService.start()` receives an optional per-generation model-path snapshot, while `JitterApp` owns the switch token and phase, coordinates movement cancellation and AI restart, and commits either the candidate or one rollback generation only after `ready`.

**Tech Stack:** Python 3.11+, Tkinter/ttk, immutable dataclasses, `pathlib`, daemon `threading`, ONNX Runtime DirectML with the existing CPU fallback, DXCam, NumPy, and `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-28-selectable-ai-model-design.md`

## Global Constraints

- Keep the centered capture and detector contract exactly `[1, 3, 320, 320]` float input `images` and `[1, 300, 6]` float output `output0`.
- Preserve class `0` as player, class `7` as head, existing preprocessing/output parsing, DirectML-first provider order, and CPU fallback.
- Every process starts with `models/all_games_320.onnx`; model path, filename, hash, and switch state are runtime-only and never enter Schema 5.
- External `.onnx` files are referenced in place. Never copy them into `models/`, build output, release materials, configuration, backups, or an application cache.
- Keep the bundled model, its fixed SHA-256, and `DmlExecutionProvider` as the only canonical `--ai-runtime-self-check` release gate.
- Add no Torch, Ultralytics, OpenCV, alternate runtime, download, training, conversion, model catalog, profile, tray behavior, or dependency.
- Keep all Tk/file-dialog/widget work on the Tk thread; model construction runs only on a daemon worker and reports through the UI queue.
- Preserve immediate STOP, disconnect, source-change, hotkey-disable, shutdown, motion-generation, target-revision, tracker, zoom, Overlay, and Test 3s behavior.
- Switching is unavailable throughout every Test 3s pending, loading, and running phase.
- Do not run Nuitka unless the user explicitly requests a packaged build.
- Use TDD for every production change: add focused failing tests, observe RED, implement minimally, rerun focused tests, and commit.

## File and Interface Map

- Create `ai_model_selection.py`: owns `ModelChoice`, `ModelSelectionError`, `ModelValidationEvent`, `bundled_model_choice()`, `external_model_choice()`, and `ModelValidator`.
- Modify `ai_service.py`: freezes an optional model path into each `start()` generation and passes it to `_worker()`.
- Modify `ui.py`: owns `_ModelSwitch`, the committed `ModelChoice`, switch token/phase, file dialog, model controls, rollback orchestration, and queue dispatch.
- Create `tests/test_ai_model_selection.py`: hardware-free path and validator tests.
- Modify `tests/test_ai_service.py`: verifies per-generation model-path snapshots and unchanged generation behavior.
- Modify `tests/test_ui.py`: adds injected chooser/validator stubs and covers idle/active switching, rollback, cancellation, stale events, Test 3s, layout, and persistence behavior.
- Modify `tests/test_settings.py`: proves an unsupported `model_path` is ignored and never written back.
- Modify `README.md` and `AGENTS.md`: documents the narrowly approved runtime model selection and adds `ai_model_selection.py` to the repository/verification maps.
- Keep `main.py`, `distribution_metadata.py`, `nuitka-package.config.yml`, `requirements.txt`, and `models/` unchanged; their existing tests are mandatory regressions.

---

### Task 1: Add immutable model choices and the one-shot validator

**Files:**
- Create: `ai_model_selection.py`
- Create: `tests/test_ai_model_selection.py`

**Interfaces:**
- Consumes: `ai_detection.model_resource_path`, `ai_detection.OnnxDetector`, a user-selected `str | Path`, and a thread-safe event sink.
- Produces:
  - `ModelChoice(path: Path, display_name: str, is_default: bool)`
  - `ModelSelectionError(ValueError)` with path-free UI-safe messages
  - `ModelValidationEvent(kind: str, token: int, choice: ModelChoice, error_type: str | None = None)`
  - `bundled_model_choice() -> ModelChoice`
  - `external_model_choice(raw_path: str | Path) -> ModelChoice`
  - `ModelValidator.start(choice: ModelChoice, token: int) -> bool`
  - `ModelValidator.cancel() -> None`
  - `ModelValidator.close() -> None`

- [ ] **Step 1: Write failing choice-validation tests**

Create `tests/test_ai_model_selection.py` with temporary-file cases that lock the public values and safe failures:

```python
import tempfile
import threading
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from ai_model_selection import (
    ModelChoice,
    ModelSelectionError,
    ModelValidationEvent,
    ModelValidator,
    bundled_model_choice,
    external_model_choice,
)


class ModelChoiceTests(unittest.TestCase):
    def test_bundled_choice_uses_resolved_default_resource(self):
        with mock.patch(
            "ai_model_selection.model_resource_path",
            return_value=Path("models/all_games_320.onnx"),
        ):
            choice = bundled_model_choice()
        self.assertEqual(choice.path.name, "all_games_320.onnx")
        self.assertTrue(choice.path.is_absolute())
        self.assertEqual(choice.display_name, "all_games_320.onnx")
        self.assertTrue(choice.is_default)
        with self.assertRaises(FrozenInstanceError):
            choice.display_name = "changed.onnx"

    def test_external_choice_accepts_case_insensitive_onnx_and_resolves_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.ONNX"
            path.write_bytes(b"model")
            choice = external_model_choice(path)
        self.assertEqual(choice.path, path.resolve())
        self.assertEqual(choice.display_name, "custom.ONNX")
        self.assertFalse(choice.is_default)

    def test_external_choice_rejects_missing_wrong_suffix_and_directory_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = root / "custom.txt"
            wrong.write_bytes(b"model")
            cases = (
                (root / "missing.onnx", "Selected model file was not found"),
                (wrong, "Select an ONNX model file"),
                (root, "Selected model must be a file"),
            )
            for raw_path, message in cases:
                with self.subTest(raw_path=raw_path):
                    with self.assertRaisesRegex(ModelSelectionError, message):
                        external_model_choice(raw_path)
```

- [ ] **Step 2: Run the new suite and verify RED**

```powershell
python -m unittest discover -s tests -p test_ai_model_selection.py -v
```

Expected: FAIL because `ai_model_selection` does not exist.

- [ ] **Step 3: Implement choices and cheap filesystem validation**

Create these definitions in `ai_model_selection.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
from typing import Any

from ai_detection import OnnxDetector, model_resource_path


LOGGER = logging.getLogger(__name__)


class ModelSelectionError(ValueError):
    """A concise model-selection failure safe to display in the UI."""


@dataclass(frozen=True)
class ModelChoice:
    path: Path
    display_name: str
    is_default: bool


@dataclass(frozen=True)
class ModelValidationEvent:
    kind: str
    token: int
    choice: ModelChoice
    error_type: str | None = None


def bundled_model_choice() -> ModelChoice:
    path = model_resource_path().resolve()
    return ModelChoice(path, path.name, True)


def external_model_choice(raw_path: str | Path) -> ModelChoice:
    path = Path(raw_path)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ModelSelectionError("Selected model file was not found") from error
    if not resolved.is_file():
        raise ModelSelectionError("Selected model must be a file")
    if resolved.suffix.lower() != ".onnx":
        raise ModelSelectionError("Select an ONNX model file")
    return ModelChoice(resolved, resolved.name, False)
```

- [ ] **Step 4: Add failing daemon, cancellation, and sanitization tests**

Append validator tests using a detector with only a `provider` attribute; never provide or call `detect()`:

```python
class ModelValidatorTests(unittest.TestCase):
    def test_validator_constructs_model_on_named_daemon_without_inference(self):
        events = []
        finished = threading.Event()
        calls = []

        class ContractOnlyDetector:
            provider = "DmlExecutionProvider"

        def detector_factory(path):
            thread = threading.current_thread()
            calls.append((path, thread.name, thread.daemon))
            return ContractOnlyDetector()

        choice = ModelChoice(Path("chosen.onnx"), "chosen.onnx", False)
        validator = ModelValidator(
            lambda event: (events.append(event), finished.set()),
            detector_factory=detector_factory,
        )
        self.addCleanup(validator.close)
        self.assertTrue(validator.start(choice, 4))
        self.assertTrue(finished.wait(1.0))
        self.assertEqual(calls, [(choice.path, "ModelValidation-4", True)])
        self.assertEqual(events, [ModelValidationEvent("ready", 4, choice)])

    def test_new_validation_cancels_late_result_from_previous_token(self):
        first_entered = threading.Event()
        release_first = threading.Event()
        first_returned = threading.Event()
        second_ready = threading.Event()
        events = []

        class Detector:
            provider = "CPUExecutionProvider"

        def detector_factory(path):
            if path.name == "first.onnx":
                first_entered.set()
                release_first.wait(1.0)
                first_returned.set()
            return Detector()

        def sink(event):
            events.append(event)
            if event.token == 2:
                second_ready.set()

        validator = ModelValidator(sink, detector_factory=detector_factory)
        self.addCleanup(validator.close)
        first = ModelChoice(Path("first.onnx"), "first.onnx", False)
        second = ModelChoice(Path("second.onnx"), "second.onnx", False)
        self.assertTrue(validator.start(first, 1))
        self.assertTrue(first_entered.wait(1.0))
        self.assertTrue(validator.start(second, 2))
        self.assertTrue(second_ready.wait(1.0))
        release_first.set()
        self.assertTrue(first_returned.wait(1.0))
        self.assertEqual(events, [ModelValidationEvent("ready", 2, second)])

    def test_failure_event_hides_exception_text_but_log_keeps_diagnostics(self):
        choice = ModelChoice(Path("secret.onnx"), "secret.onnx", False)
        events = []
        ready = threading.Event()

        def fail(_path):
            raise RuntimeError("sensitive absolute path detail")

        validator = ModelValidator(
            lambda event: (events.append(event), ready.set()),
            detector_factory=fail,
        )
        self.addCleanup(validator.close)
        with self.assertLogs("ai_model_selection", level="ERROR") as logs:
            self.assertTrue(validator.start(choice, 9))
            self.assertTrue(ready.wait(1.0))
        self.assertEqual(
            events,
            [ModelValidationEvent("error", 9, choice, "RuntimeError")],
        )
        self.assertNotIn("sensitive absolute path detail", repr(events))
        self.assertIn("sensitive absolute path detail", "\n".join(logs.output))

    def test_thread_start_failure_returns_false_without_duplicate_error_event(self):
        class FailingThread:
            def __init__(self, **_kwargs):
                pass

            def start(self):
                raise RuntimeError("scheduler detail")

        events = []
        validator = ModelValidator(events.append)
        self.addCleanup(validator.close)
        choice = ModelChoice(Path("chosen.onnx"), "chosen.onnx", False)
        with (
            mock.patch("ai_model_selection.threading.Thread", FailingThread),
            self.assertLogs("ai_model_selection", level="ERROR") as logs,
        ):
            self.assertFalse(validator.start(choice, 3))
        self.assertEqual(events, [])
        self.assertIn("scheduler detail", "\n".join(logs.output))
```

- [ ] **Step 5: Implement the generation-safe validator**

Add `ModelValidator` with a short state lock and event barrier. `start()` first signals the previous stop event, installs `(token, stop_event)`, then starts a `ModelValidation-{token}` daemon. `_worker()` constructs `self._detector_factory(choice.path)`, reads `provider` to force completed session setup, checks currency, and emits exactly one `ready`; on exception it logs with `LOGGER.exception("AI model validation failed for %s", choice.path)` and emits the sanitized exception type. `cancel()` sets and clears the active stop event, while `close()` also makes later `start()` calls return `False`.

```python
class ModelValidator:
    def __init__(
        self,
        event_sink: Callable[[ModelValidationEvent], None],
        detector_factory: Callable[[Path | str], Any] = OnnxDetector,
    ) -> None:
        self._event_sink = event_sink
        self._detector_factory = detector_factory
        self._lock = threading.Lock()
        self._event_lock = threading.RLock()
        self._active: tuple[int, threading.Event] | None = None
        self._closed = False

    def start(self, choice: ModelChoice, token: int) -> bool:
        with self._lock:
            if self._closed:
                return False
            if self._active is not None:
                self._active[1].set()
            stop_event = threading.Event()
            self._active = (token, stop_event)
        try:
            threading.Thread(
                target=self._worker,
                args=(choice, token, stop_event),
                name=f"ModelValidation-{token}",
                daemon=True,
            ).start()
        except Exception as error:
            stop_event.set()
            with self._lock:
                if self._active == (token, stop_event):
                    self._active = None
            LOGGER.exception("AI model validation worker could not start")
            return False
        return True

    def _worker(
        self,
        choice: ModelChoice,
        token: int,
        stop_event: threading.Event,
    ) -> None:
        try:
            if stop_event.is_set():
                return
            detector = self._detector_factory(choice.path)
            _provider = detector.provider
            event = ModelValidationEvent("ready", token, choice)
        except Exception as error:
            LOGGER.exception("AI model validation failed for %s", choice.path)
            event = ModelValidationEvent(
                "error", token, choice, type(error).__name__
            )
        try:
            self._emit_current(event, token, stop_event)
        finally:
            with self._lock:
                if self._active == (token, stop_event):
                    self._active = None

    def _emit_current(
        self,
        event: ModelValidationEvent,
        token: int,
        stop_event: threading.Event,
    ) -> None:
        with self._event_lock:
            with self._lock:
                current = (
                    not self._closed
                    and not stop_event.is_set()
                    and self._active == (token, stop_event)
                )
            if current:
                self._event_sink(event)

    def cancel(self) -> None:
        with self._lock:
            if self._active is not None:
                self._active[1].set()
                self._active = None
        with self._event_lock:
            pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._active is not None:
                self._active[1].set()
                self._active = None
        with self._event_lock:
            pass
```

Add a patched-`threading.Thread` test asserting a constructor/start exception is logged, `start()` returns `False`, and no event is emitted. The UI owns the one resulting rollback decision; the validator must not also enqueue an error. A stopped/obsolete worker must never call the sink.

- [ ] **Step 6: Run focused tests and commit**

```powershell
python -m unittest discover -s tests -p test_ai_model_selection.py -v
git diff --check
git add ai_model_selection.py tests/test_ai_model_selection.py
git commit -m "feat: add runtime AI model validation"
```

Expected: all model-selection tests PASS; no DXCam factory or inference method is used.

---

### Task 2: Freeze the model path into each AI service generation

**Files:**
- Modify: `ai_service.py`
- Modify: `tests/test_ai_service.py`

**Interfaces:**
- Consumes: the Task 1 `ModelChoice.path` as `Path | str`.
- Produces: `AiService.start(settings_provider, zoom_gate_provider=None, *, model_path=None) -> int | None`; the chosen path is an immutable argument to `_worker()` for that generation.

- [ ] **Step 1: Add failing per-generation path tests**

Add tests beside `test_explicit_model_path_is_created_on_daemon_worker`:

```python
def test_start_model_path_overrides_constructor_default_for_one_generation(self):
    observed = []
    ready = threading.Event()

    def detector_factory(path):
        observed.append(path)
        return FakeDetector()

    service = AiService(
        lambda event: ready.set() if event.kind == "ready" else None,
        model_path="constructor.onnx",
        detector_factory=detector_factory,
        capture_factory=FakeCapture,
    )
    self.addCleanup(service.close)
    service.start(AimSettings, model_path=Path("generation.onnx"))
    self.assertTrue(ready.wait(1.0))
    self.assertEqual(observed, [Path("generation.onnx")])

def test_running_generation_keeps_first_path_and_restart_uses_second_path(self):
    paths = []
    ready_count = threading.Event()

    def detector_factory(path):
        paths.append(path)
        if len(paths) == 2:
            ready_count.set()
        return FakeDetector()

    service = AiService(
        lambda _event: None,
        detector_factory=detector_factory,
        capture_factory=FakeCapture,
    )
    self.addCleanup(service.close)
    first = service.start(AimSettings, model_path="first.onnx")
    duplicate = service.start(AimSettings, model_path="ignored.onnx")
    self.assertEqual(duplicate, first)
    service.stop("switch")
    service.start(AimSettings, model_path="second.onnx")
    self.assertTrue(ready_count.wait(1.0))
    self.assertEqual(paths, ["first.onnx", "second.onnx"])
```

- [ ] **Step 2: Run the AI service suite and verify RED**

```powershell
python -m unittest discover -s tests -p test_ai_service.py -v
```

Expected: FAIL because `AiService.start()` does not accept `model_path`.

- [ ] **Step 3: Pass a generation-local path into the worker**

Change only the service boundary and worker argument; retain the constructor option for existing callers/tests:

```python
def start(
    self,
    settings_provider: Callable[[], AimSettings],
    zoom_gate_provider: Callable[[], bool] | None = None,
    *,
    model_path: Path | str | None = None,
) -> int | None:
```

Keep the current validation, lock-protected generation allocation, snapshot
clear, `loading` status, and loading event unchanged. Replace only the worker
construction with:

```python
    worker = threading.Thread(
        target=self._worker,
        args=(
            generation,
            stop_event,
            settings_provider,
            zoom_gate_provider,
            model_path,
        ),
        name=f"AiInference-{generation}",
        daemon=True,
    )
```

Update `_worker(..., generation_model_path)` and resolve exactly once after the initial currency check:

```python
model_path = (
    generation_model_path
    if generation_model_path is not None
    else self._model_path or model_resource_path()
)
detector = self._detector_factory(model_path)
```

Do not read a mutable UI choice from inside `_worker()` and do not move default `model_resource_path()` resolution onto the caller thread.

- [ ] **Step 4: Run focused regressions and commit**

```powershell
python -m unittest discover -s tests -p test_ai_service.py -v
python -m unittest discover -s tests -p test_ai_detection.py -v
git diff --check
git add ai_service.py tests/test_ai_service.py
git commit -m "feat: select AI model per service generation"
```

Expected: both suites PASS, including cancellation before default path resolution and detector construction.

---

### Task 3: Add the model row and idle selection flow

**Files:**
- Modify: `ui.py`
- Modify: `tests/test_ui.py`

**Interfaces:**
- Consumes: Task 1 model-choice/validator interfaces and Task 2 `AiService.start(..., model_path=...)`.
- Produces:
  - `_ModelSwitch(token, candidate, previous, phase)` where phase is `validating`, `starting_candidate`, or `starting_rollback`
  - `JitterApp.browse_ai_model() -> None`
  - `JitterApp.use_default_ai_model() -> None`
  - `JitterApp.queue_model_validation_event(event) -> None`
  - `JitterApp.handle_model_validation_event(event) -> None`
  - `JitterApp._begin_model_switch(candidate) -> None`
  - `JitterApp._finish_model_switch(choice, footer) -> None`
  - `JitterApp._render_model_controls() -> None`

- [ ] **Step 1: Extend the UI fixture with a controllable validator and chooser**

Add imports for `ModelChoice` and `ModelValidationEvent`, then add:

```python
class StubModelValidator:
    def __init__(self):
        self.event_sink = None
        self.start_calls = []
        self.cancelled = 0
        self.closed = 0
        self.start_result = True

    def with_sink(self, event_sink):
        self.event_sink = event_sink
        return self

    def start(self, choice, token):
        self.start_calls.append((choice, token))
        return self.start_result

    def cancel(self):
        self.cancelled += 1

    def close(self):
        self.closed += 1

    def emit(self, event):
        self.event_sink(event)
```

In `make_app()`, create `self.model_validator`, `self.model_dialog_result`, and inject:

```python
model_validator_factory=lambda sink: self.model_validator.with_sink(sink),
model_file_chooser=lambda **_kwargs: self.model_dialog_result,
```

Update `StubAiService.start()` to accept `*, model_path=None` and append `(settings_provider, zoom_gate_provider, model_path)` so every UI test can assert the exact generation model.

Update the one pre-existing tuple-unpack assertion so the regression still
checks the same zoom provider and also sees the bundled path:

```python
_settings_provider, zoom_provider, model_path = self.ai.start_calls[-1]
self.assertEqual(model_path, self.app._model_choice.path)
self.assertIs(zoom_provider.__self__, self.app)
```

Add this fixture helper to keep selected files alive for the whole test:

```python
def begin_custom_model_switch(self, filename):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    path = Path(temporary.name) / filename
    path.write_bytes(b"model")
    self.model_dialog_result = str(path)
    self.app.browse_ai_model()
    return self.model_validator.start_calls[-1]
```

- [ ] **Step 2: Add failing layout, chooser, idle commit, and no-save tests**

Use a temporary real `.onnx` file and drive validator events manually:

```python
def test_model_row_starts_with_bundled_default_and_keeps_fixed_shell(self):
    self.assertEqual(
        self.app.ai_model_var.get(),
        "Default · all_games_320.onnx",
    )
    self.assertEqual(str(self.app.use_default_model_button.cget("state")), "disabled")
    self.app.update_idletasks()
    self.assertEqual(self.app.geometry().split("+")[0], "840x620")
    self.assertEqual(self.app.stop_button.winfo_manager(), "grid")

def test_browse_cancel_does_not_change_model_or_schedule_save(self):
    self.app._cancel_after("_save_after_id")
    self.app.model_browse_button.invoke()
    self.assertEqual(self.model_validator.start_calls, [])
    self.assertEqual(self.app.ai_model_var.get(), "Default · all_games_320.onnx")
    self.assertIsNone(self.app._save_after_id)

def test_idle_candidate_commits_after_matching_validation_ready(self):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "custom.onnx"
        path.write_bytes(b"model")
        self.model_dialog_result = str(path)
        self.app.model_browse_button.invoke()
        choice, token = self.model_validator.start_calls[-1]
        self.assertEqual(self.app.ai_model_var.get(), "Loading · custom.onnx")
        self.assertEqual(str(self.app.model_browse_button.cget("state")), "disabled")
        self.model_validator.emit(ModelValidationEvent("ready", token, choice))
        self.drain_ui_queue()
    self.assertEqual(self.app.ai_model_var.get(), "Custom · custom.onnx")
    self.assertEqual(self.ai.start_calls, [])
    self.assertIsNone(self.app._save_after_id)

def test_use_default_runs_the_same_validation_flow_without_saving(self):
    custom, custom_token = self.begin_custom_model_switch("custom.onnx")
    self.model_validator.emit(
        ModelValidationEvent("ready", custom_token, custom)
    )
    self.drain_ui_queue()
    self.app._cancel_after("_save_after_id")

    self.app.use_default_model_button.invoke()
    default, default_token = self.model_validator.start_calls[-1]
    self.assertTrue(default.is_default)
    self.assertEqual(self.app.ai_model_var.get(), "Loading · all_games_320.onnx")
    self.model_validator.emit(
        ModelValidationEvent("ready", default_token, default)
    )
    self.drain_ui_queue()

    self.assertEqual(self.app.ai_model_var.get(), "Default · all_games_320.onnx")
    self.assertIsNone(self.app._save_after_id)
```

Also add wrong-suffix and missing-path cases asserting the previous label/service stay unchanged and the footer contains no absolute path.

- [ ] **Step 3: Run the UI suite and verify RED**

```powershell
python -m unittest discover -s tests -p test_ui.py -v
```

Expected: FAIL because the model UI, injected seams, and event queue branch do not exist.

- [ ] **Step 4: Add runtime model state and the scrollable MODEL row**

Import `filedialog` and Task 1 interfaces. Define the UI-owned record beside `_DeferredMotionAction`:

```python
@dataclass(frozen=True)
class _ModelSwitch:
    token: int
    candidate: ModelChoice
    previous: ModelChoice
    phase: str
```

Extend `JitterApp.__init__` with optional `model_validator_factory` and `model_file_chooser`. Before `_create_variables()`, initialize the bundled committed choice, monotonic token, and empty switch. After widgets exist, build the validator with `queue_model_validation_event`. `_create_variables()` creates `ai_model_var` from `_model_label(self._model_choice)`.

Add a full-width row below Target Area in `ai_settings_card`:

```python
self.ai_model_frame = ttk.Frame(
    self.ai_settings_card, style="Liquid.Surface.TFrame", padding=(5, 10)
)
self.ai_model_frame.grid(row=4, column=0, sticky="ew")
self.ai_model_frame.columnconfigure(0, weight=1)
ttk.Label(
    self.ai_model_frame, text="MODEL", style="Liquid.CardBody.TLabel"
).grid(row=0, column=0, columnspan=2, sticky="w")
ttk.Label(
    self.ai_model_frame,
    textvariable=self.ai_model_var,
    style="Liquid.CardText.TLabel",
    wraplength=300,
).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 6))
self.model_browse_button = ttk.Button(
    self.ai_model_frame,
    text="Browse...",
    style="Liquid.Secondary.TButton",
    command=self.browse_ai_model,
)
self.model_browse_button.grid(row=2, column=0, sticky="ew", padx=(0, 3))
self.use_default_model_button = ttk.Button(
    self.ai_model_frame,
    text="Use Default",
    style="Liquid.Secondary.TButton",
    command=self.use_default_ai_model,
)
self.use_default_model_button.grid(row=2, column=1, sticky="ew", padx=(3, 0))
```

The dialog call must use `title="Select AI Aim ONNX Model"`, `filetypes=(("ONNX models", "*.onnx"), ("All files", "*.*"))`, and `parent=self`. Only pass a nonempty result through `external_model_choice()`; log the raw path on validation failure but show only `str(error)` in the footer.

- [ ] **Step 5: Queue validation events and commit idle choices**

Add a `"model_validation"` branch in `_drain_ui_queue()` and keep Tk state out of the worker sink:

```python
@staticmethod
def _model_label(choice: ModelChoice) -> str:
    prefix = "Default" if choice.is_default else "Custom"
    return f"{prefix} · {choice.display_name}"

def _begin_model_switch(self, candidate: ModelChoice) -> None:
    if candidate.path == self._model_choice.path:
        self.footer_var.set(f"Using model: {candidate.display_name}")
        return
    self._model_switch_token += 1
    switch = _ModelSwitch(
        self._model_switch_token,
        candidate,
        self._model_choice,
        "validating",
    )
    self._model_switch = switch
    self.ai_model_var.set(f"Loading · {candidate.display_name}")
    self._render_model_controls()
    if not self.model_validator.start(candidate, switch.token):
        self._finish_model_switch(
            switch.previous,
            f"Model rejected; restored {switch.previous.display_name}",
        )

def queue_model_validation_event(self, event: ModelValidationEvent) -> None:
    if self._closing or self._closed:
        return
    self._ui_queue.put(("model_validation", None, event))

def handle_model_validation_event(self, event: ModelValidationEvent) -> None:
    switch = self._model_switch
    if switch is None or event.token != switch.token or event.choice != switch.candidate:
        return
    if event.kind == "error":
        self._finish_model_switch(
            switch.previous,
            f"Model rejected; restored {switch.previous.display_name}",
        )
        return
    if event.kind != "ready":
        return
    if not self._ai_runtime_required():
        self._finish_model_switch(
            switch.candidate,
            f"Using model: {switch.candidate.display_name}",
        )
        return
    self._finish_model_switch(
        switch.previous,
        "Model change cancelled because AI started",
    )
```

For this independently safe UI/idle commit, `_render_model_controls()` also
disables model changes while `_ai_runtime_required()` is true, and both command
handlers repeat that guard with `Stop AI before changing the model`. Task 4
removes that temporary restriction when generation switching is implemented.
If AI demand is added while idle validation is already running, the handler
finishes with `switch.previous` and reports `Model change cancelled because AI
started`; it does not start AI. `_finish_model_switch()`
commits the supplied choice, clears the record, refreshes the label/buttons,
and sets the footer without scheduling save.

- [ ] **Step 6: Run focused tests and commit**

```powershell
python -m unittest discover -s tests -p test_ai_model_selection.py -v
python -m unittest discover -s tests -p test_ui.py -v
git diff --check
git add ui.py tests/test_ui.py
git commit -m "feat: add runtime AI model controls"
```

Expected: idle selection is validated off-thread, the absolute path never appears in a widget, and the fixed shell/STOP tests pass.

---

### Task 4: Implement active switching and one-level rollback

**Files:**
- Modify: `ui.py`
- Modify: `tests/test_ui.py`

**Interfaces:**
- Consumes: current `_ai_runtime_required()`, Task 2 per-start path, AI `ready/error/stopped` events, and `_ModelSwitch`.
- Produces:
  - active-demand behavior in the Task 3 `_begin_model_switch(candidate)` flow
  - `_start_validated_model_generation(switch: _ModelSwitch) -> None`
  - `_start_model_rollback(switch: _ModelSwitch, failure: str) -> None`
  - `_handle_ai_runtime_error(payload: object) -> None` for the unchanged fail-closed terminal path

- [ ] **Step 1: Add failing active-success tests**

Cover AI-only, combined, and Overlay-only demand. The central active case is:

```python
def test_active_switch_stops_motion_and_ai_then_restarts_candidate_on_validation(self):
    self.prepare_armed_sources(MotionSources(True, True), gate_active=True)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "custom.onnx"
        path.write_bytes(b"model")
        self.model_dialog_result = str(path)
        self.app.model_browse_button.invoke()
        choice, token = self.model_validator.start_calls[-1]

        self.assertIn("model_switch", self.service.cancel_reasons)
        self.assertEqual(self.ai.stop_calls[-1], "Model switch")
        self.assertEqual(self.ai.reset_targeting_calls, 1)
        self.assertFalse(self.app._ai_ready)
        self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")

        self.model_validator.emit(ModelValidationEvent("ready", token, choice))
        self.drain_ui_queue()

    self.assertEqual(self.ai.start_calls[-1][2], choice.path)
    self.assertEqual(self.app.ai_model_var.get(), "Loading · custom.onnx")
    self.assertEqual(self.app._model_switch.phase, "starting_candidate")
    self.assertFalse(self.app._normal_motion_started)

    self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
    self.assertEqual(self.app.ai_model_var.get(), "Custom · custom.onnx")
    self.assertIsNone(self.app._model_switch)
    self.assertTrue(self.app._ai_ready)
```

Assert Master/source/Overlay selections are unchanged across the switch and that motion restarts only after candidate `ready` and the current Trigger/Modifier gate permits it.

- [ ] **Step 2: Add failing validation/runtime rollback tests**

Add these exact state transitions:

```python
def test_validation_failure_restarts_previous_model_when_demand_still_exists(self):
    self.prepare_armed_sources(MotionSources(False, True))
    choice, token = self.begin_custom_model_switch("bad.onnx")
    self.model_validator.emit(
        ModelValidationEvent("error", token, choice, "ModelContractError")
    )
    self.drain_ui_queue()
    self.assertEqual(self.ai.start_calls[-1][2], self.app._model_choice.path)
    self.assertEqual(self.app._model_switch.phase, "starting_rollback")
    self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
    self.assertIsNone(self.app._model_switch)
    self.assertTrue(self.app.footer_var.get().startswith("Model rejected; restored "))

def test_candidate_runtime_error_rolls_back_once(self):
    self.prepare_armed_sources(MotionSources(False, True))
    choice, token = self.begin_custom_model_switch("loads-then-fails.onnx")
    self.model_validator.emit(ModelValidationEvent("ready", token, choice))
    self.drain_ui_queue()
    self.app.handle_ai_event(AiEvent("error", "RuntimeError: AI service failed"))
    self.assertEqual(self.app._model_switch.phase, "starting_rollback")
    self.assertEqual(self.ai.start_calls[-1][2], self.app._model_switch.previous.path)

def test_rollback_error_enters_existing_fail_closed_path_without_retry(self):
    self.prepare_armed_sources(MotionSources(False, True))
    choice, token = self.begin_custom_model_switch("bad-runtime.onnx")
    self.model_validator.emit(ModelValidationEvent("ready", token, choice))
    self.drain_ui_queue()
    self.app.handle_ai_event(AiEvent("error", "candidate failed"))
    starts_before_failure = len(self.ai.start_calls)
    with self.assertLogs(level="ERROR"):
        self.app.handle_ai_event(AiEvent("error", "rollback failed"))
    self.assertEqual(len(self.ai.start_calls), starts_before_failure)
    self.assertIsNone(self.app._model_switch)
    self.assertFalse(self.app.ai_selected)
    self.assertFalse(self.app.master_armed)

def test_validation_thread_start_failure_restarts_previous_model_once(self):
    self.prepare_armed_sources(MotionSources(False, True))
    self.model_validator.start_result = False
    self.begin_custom_model_switch("thread-fails.onnx")
    self.assertEqual(self.app._model_switch.phase, "starting_rollback")
    self.assertEqual(self.ai.start_calls[-1][2], self.app._model_switch.previous.path)
```

Use the Task 3 `begin_custom_model_switch()` fixture helper for all rollback cases.

- [ ] **Step 3: Run the UI suite and verify RED**

```powershell
python -m unittest discover -s tests -p test_ui.py -v
```

Expected: idle tests pass; active restart/rollback assertions fail.

- [ ] **Step 4: Route every AI start through the committed or pending model snapshot**

Remove Task 3's idle-only `_ai_runtime_required()` control/command guard. Change
the validation-ready demand branch to call
`_start_validated_model_generation(switch)`, then change `_start_ai_runtime` to
accept a choice and pass the exact path:

```python
def _start_ai_runtime(
    self,
    context: str,
    *,
    model_choice: ModelChoice | None = None,
) -> bool:
    choice = model_choice or self._model_choice
    if not self._ai_runtime_active:
        self._ai_event_epoch += 1
    self._ai_runtime_active = True
    self._sync_adaptive_zoom_gate()
    try:
        generation = self.ai_service.start(
            self.get_ai_settings,
            self.get_adaptive_zoom_gate,
            model_path=choice.path,
        )
    except Exception:
        logging.exception("AI runtime could not start during %s", context)
        self._ai_ready = False
        self._ai_provider = None
        self._ai_runtime_active = False
        self._sync_adaptive_zoom_gate()
        self.ai_status_var.set("Error")
        self.ai_fps_var.set("0 FPS")
        self.ai_provider_var.set("No provider")
        return False
    if not generation:
        logging.error("AI runtime did not start during %s", context)
        self._ai_ready = False
        self._ai_provider = None
        self._ai_runtime_active = False
        self._sync_adaptive_zoom_gate()
        self.ai_status_var.set("Error")
        self.ai_fps_var.set("0 FPS")
        self.ai_provider_var.set("No provider")
        return False
    return True
```

Extend Task 3's `_begin_model_switch()` before `model_validator.start()`:

```python
if self._normal_motion_started or self._expected_motion_generation is not None:
    self._stop_motion_runtime("model_switch")
    self._normal_motion_started = False
    self._set_runtime_state("armed" if self.master_armed else "disabled")
self._ai_targeting_revision = self.ai_service.reset_targeting()
self.ai_zoom_var.set("1.0×")
if self._ai_runtime_active:
    self._stop_ai_runtime("Model switch")
self._ai_ready = False
self._ai_provider = None
self._ai_runtime_active = False
self._sync_adaptive_zoom_gate()
self.ai_status_var.set("Loading")
self.ai_fps_var.set("0 FPS")
self.ai_provider_var.set("No provider")
```

Preserve `master_armed`, `jitter_selected`, `ai_selected`, and
`overlay_visible`. Replace the Task 3 validation-ready demand cancellation
with these exact candidate/rollback helpers:

```python
def _start_validated_model_generation(self, switch: _ModelSwitch) -> None:
    if self._model_switch != switch or switch.phase != "validating":
        return
    starting = replace(switch, phase="starting_candidate")
    self._model_switch = starting
    self._render_model_controls()
    if not self._start_ai_runtime(
        "Model switch", model_choice=starting.candidate
    ):
        self._start_model_rollback(starting, "candidate startup failed")

def _start_model_rollback(
    self,
    switch: _ModelSwitch,
    failure: str,
) -> None:
    if self._model_switch != switch:
        return
    logging.error(
        "AI model %s rejected: %s",
        switch.candidate.path,
        failure,
    )
    if self._ai_runtime_active:
        self._stop_ai_runtime("Model rollback")
    self._ai_ready = False
    self._ai_provider = None
    self._ai_runtime_active = False
    self._sync_adaptive_zoom_gate()
    if not self._ai_runtime_required():
        self._finish_model_switch(
            switch.previous,
            f"Model rejected; restored {switch.previous.display_name}",
        )
        return
    rollback = replace(switch, phase="starting_rollback")
    self._model_switch = rollback
    self.ai_model_var.set(f"Loading · {rollback.previous.display_name}")
    self._render_model_controls()
    if self._start_ai_runtime(
        "Model rollback", model_choice=rollback.previous
    ):
        return
    self._model_switch = None
    self._render_model_controls()
    self._handle_ai_runtime_error("AI model rollback failed")
```

Call `_start_model_rollback(switch, event.error_type or "validation failed")`
for validator `error`; call `_start_validated_model_generation(switch)` for a
matching validator `ready` when current demand is true. Replace Task 3's
falsey validator-start branch with:

```python
if not self.model_validator.start(candidate, switch.token):
    self._start_model_rollback(switch, "validation worker could not start")
```

- [ ] **Step 5: Commit on ready and roll back exactly once on error**

At the beginning of the existing `ready` branch:

```python
switch = self._model_switch
if switch is not None and switch.phase == "starting_candidate":
    self._finish_model_switch(
        switch.candidate,
        f"Using model: {switch.candidate.display_name}",
    )
elif switch is not None and switch.phase == "starting_rollback":
    self._finish_model_switch(
        switch.previous,
        f"Model rejected; restored {switch.previous.display_name}",
    )
```

Extract the current long `error` body verbatim into `_handle_ai_runtime_error(payload)` and make the `error` branch dispatch by phase:

```python
switch = self._model_switch
if switch is not None and switch.phase == "starting_candidate":
    failure = str(event.payload or "AI service failed")
    self._start_model_rollback(switch, failure)
    return
if switch is not None and switch.phase == "starting_rollback":
    self._model_switch = None
    self._render_model_controls()
    self._handle_ai_runtime_error(event.payload)
    return
self._handle_ai_runtime_error(event.payload)
```

Validation failure uses the same `_start_model_rollback()`. That method restores immediately without a service start when current demand is false; otherwise it sets `starting_rollback` before calling `_start_ai_runtime(..., model_choice=switch.previous)`. A synchronous/falsey rollback start calls `_handle_ai_runtime_error()` directly and never calls `_start_model_rollback()` again.

- [ ] **Step 6: Run lifecycle regressions and commit**

```powershell
python -m unittest discover -s tests -p test_ui.py -v
python -m unittest discover -s tests -p test_ai_service.py -v
python -m unittest discover -s tests -p test_combined_motion.py -v
git diff --check
git add ui.py tests/test_ui.py
git commit -m "feat: switch AI model generations safely"
```

Expected: active success/rollback tests PASS; the pre-existing AI error fallback tests remain unchanged and PASS.

---

### Task 5: Close cancellation races and Test 3s control states

**Files:**
- Modify: `ui.py`
- Modify: `tests/test_ui.py`

**Interfaces:**
- Consumes: the Task 4 switch state and all existing lifecycle entry points.
- Produces: `_cancel_model_switch(reason: str) -> None`, which increments the switch token, signals validator cancellation, restores the previous committed choice, clears pending auto-restart, and never changes persisted data.

- [ ] **Step 1: Add failing stale-event and cancellation tests**

Add a table-driven test that begins a switch, invokes each lifecycle action, then submits the old validator event:

```python
def test_stop_invalidates_switch_before_late_validation_ready(self):
    self.app.toggle_overlay()
    choice, token = self.begin_custom_model_switch("late.onnx")
    starts = len(self.ai.start_calls)
    self.app.emergency_stop("Stopped by user")
    self.model_validator.emit(ModelValidationEvent("ready", token, choice))
    self.drain_ui_queue()
    self.assertIsNone(self.app._model_switch)
    self.assertEqual(len(self.ai.start_calls), starts)
    self.assertEqual(self.app.ai_model_var.get(), "Default · all_games_320.onnx")
    self.assertFalse(self.app.overlay_visible)

def test_stale_ready_from_cancelled_switch_cannot_commit_after_new_switch(self):
    first, first_token = self.begin_custom_model_switch("first.onnx")
    self.app.emergency_stop()
    second, second_token = self.begin_custom_model_switch("second.onnx")
    self.model_validator.emit(ModelValidationEvent("ready", first_token, first))
    self.drain_ui_queue()
    self.assertEqual(self.app._model_switch.token, second_token)
    self.assertEqual(self.app.ai_model_var.get(), "Loading · second.onnx")
```

Add equivalent assertions for `close_app`, `_handle_disconnect`, Master/hotkey
disable when it removes final AI demand, removing the AI source, and hiding
Overlay when that leaves no demand. Include a case where Overlay remains
visible after AI-source removal and verify the previous model restarts from
current Overlay demand rather than a captured old demand flag.

- [ ] **Step 2: Add failing Test 3s and control-state tests**

```python
def test_model_controls_are_disabled_for_every_test_mode(self):
    modes = (
        "test_jitter_pending",
        "test_jitter",
        "test_ai_loading",
        "test_ai",
        "test_combined_loading",
        "test_combined",
    )
    for mode in modes:
        with self.subTest(mode=mode):
            self.app._motion_mode = mode
            self.app._render_runtime_controls()
            self.assertEqual(str(self.app.model_browse_button.cget("state")), "disabled")
            self.assertEqual(str(self.app.use_default_model_button.cget("state")), "disabled")
            self.assertEqual(str(self.app.stop_button.cget("state")), "normal")

def test_browse_handler_refuses_direct_call_during_test(self):
    self.app._motion_mode = "test_ai_loading"
    self.app.browse_ai_model()
    self.assertEqual(self.model_validator.start_calls, [])
    self.assertEqual(self.app.footer_var.get(), "Test Run is active; use STOP to cancel")

def test_busy_switch_rejects_repeated_browse_and_default_commands(self):
    choice, token = self.begin_custom_model_switch("first.onnx")
    calls = list(self.model_validator.start_calls)
    self.app.browse_ai_model()
    self.app.use_default_ai_model()
    self.assertEqual(self.model_validator.start_calls, calls)
    self.assertEqual(self.app._model_switch.token, token)
    self.assertEqual(self.app._model_switch.candidate, choice)
```

Also assert model controls are disabled throughout validation/candidate startup/rollback and re-enabled only after commit or terminal failure; STOP remains `normal` in each phase.

- [ ] **Step 3: Run the UI suite and verify RED**

```powershell
python -m unittest discover -s tests -p test_ui.py -v
```

Expected: active switching passes; stale/cancellation and Test 3s cases fail until every entry point invalidates correctly.

- [ ] **Step 4: Implement token invalidation before service reconciliation**

Use one helper and call it before lifecycle code can stop or restart AI:

```python
def _cancel_model_switch(self, reason: str) -> None:
    switch = self._model_switch
    if switch is None:
        return
    self._model_switch_token += 1
    self.model_validator.cancel()
    if switch.phase in {"starting_candidate", "starting_rollback"}:
        self._stop_ai_runtime("Model switch cancelled")
        self._ai_ready = False
        self._ai_provider = None
        self._ai_runtime_active = False
        self._sync_adaptive_zoom_gate()
    self._model_switch = None
    self._model_choice = switch.previous
    self.ai_model_var.set(self._model_label(self._model_choice))
    self._render_model_controls()
    logging.info("AI model switch cancelled: %s", reason)
```

Call it after the relevant boolean state is updated but before service
reconciliation in `emergency_stop()`, `_handle_disconnect()`, `close_app()`,
Master/hotkey disable when it removes final demand, AI-source removal, and
Overlay removal when `_ai_runtime_required()` becomes false. This ordering
lets `_cancel_model_switch()` stop a pending candidate, then lets
`_reconcile_ai_runtime()` read the new current demand. After AI-source removal
with Overlay still visible, reconciliation restarts the previous model for
Overlay demand. Do not cache demand in `_ModelSwitch`.

In `close_app()`, call `model_validator.close()` after invalidating the token and before destroying Tk. Late worker sink calls are dropped by both validator currency and `queue_model_validation_event()` closing checks.

- [ ] **Step 5: Centralize model-control rendering**

At the end of `_render_runtime_controls()`, call `_render_model_controls()`. Its state rules are exact:

```python
busy = self._model_switch is not None
testing = self._motion_mode in _TEST_MOTION_MODES
disabled = busy or testing or self._closing
self.model_browse_button.configure(state="disabled" if disabled else "normal")
self.use_default_model_button.configure(
    state=(
        "disabled"
        if disabled or self._model_choice.is_default
        else "normal"
    )
)
```

Both command handlers repeat the Test 3s/busy guard so direct method calls cannot bypass widget state.

- [ ] **Step 6: Run focused cancellation regressions and commit**

```powershell
python -m unittest discover -s tests -p test_ui.py -v
python -m unittest discover -s tests -p test_ai_service.py -v
python -m unittest discover -s tests -p test_makcu_service.py -v
git diff --check
git add ui.py tests/test_ui.py
git commit -m "fix: cancel stale AI model switches"
```

Expected: old validator/AI events neither commit a model nor restart inference after cancellation; STOP and existing motion barriers PASS.

---

### Task 6: Lock runtime-only configuration, packaging, and documentation

**Files:**
- Modify: `tests/test_settings.py`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Verify unchanged: `main.py`
- Verify unchanged: `distribution_metadata.py`
- Verify unchanged: `nuitka-package.config.yml`
- Verify unchanged: `requirements.txt`

**Interfaces:**
- Consumes: Schema 5 `ConfigStore`, canonical self-check metadata, and the completed UI behavior.
- Produces: documentation and regression evidence that only the bundled model participates in startup defaults, persistence, packaging, and release self-check.

- [ ] **Step 1: Add a failing/locking Schema 5 omission test**

Add beside `test_schema_five_round_trips_ai_settings_without_mode`:

```python
def test_schema_five_ignores_and_never_rewrites_model_selection(self):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps({
            "schema_version": 5,
            "model_path": "C:/private/custom.onnx",
            "ai": {"model_path": "nested.onnx"},
        }), encoding="utf-8")
        store = ConfigStore(path)
        outcome = store.load()
        self.assertFalse(hasattr(outcome.config, "model_path"))
        store.save(outcome.config)
        document = json.loads(path.read_text(encoding="utf-8"))
    self.assertNotIn("model_path", document)
    self.assertNotIn("model_path", document["ai"])
    self.assertEqual(document["schema_version"], 5)

def test_schema_five_generated_config_and_backup_never_contain_model_data(self):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        store = ConfigStore(path)
        store.save(AppConfig(theme="dark"))
        store.save(AppConfig(theme="light"))
        current = json.loads(path.read_text(encoding="utf-8"))
        backup = json.loads(
            path.with_name("config.json.bak").read_text(encoding="utf-8")
        )
    for document in (current, backup):
        self.assertNotIn("model_path", document)
        self.assertNotIn("model_name", document)
        self.assertNotIn("model_hash", document)
        self.assertNotIn("model_path", document["ai"])
```

- [ ] **Step 2: Run persistence and packaging regressions**

```powershell
python -m unittest discover -s tests -p test_settings.py -v
python -m unittest discover -s tests -p test_entrypoints.py -v
python -m unittest discover -s tests -p test_distribution_metadata.py -v
```

Expected: PASS without changing schema, self-check hash/path, package data options, or dependency metadata.

- [ ] **Step 3: Document the approved narrow model choice**

In `README.md`, replace “It does not support arbitrary models” with a concise MODEL-row section that states:

```markdown
The `MODEL` row starts with `Default · all_games_320.onnx`. `Browse...` can
select an external `.onnx` file for this process only, and `Use Default`
returns to the bundled model. A custom model must keep the exact `images`
`[1,3,320,320]` and `output0` `[1,300,6]` float contract and use class 0 for
players and class 7 for heads. Jitter validates it off the UI thread, pauses
AI during the switch, and restores the previous model if validation or startup
fails. The selected path is never saved, copied, packaged, or used by the
release self-check; every launch starts with the bundled model.
```

In `AGENTS.md`, add `ai_model_selection.py` to Planned repository layout, amend the fixed-model scope bullet to allow only runtime browsing of contract-compatible external ONNX files, and add the file to the documented `py_compile` command. Keep all prohibitions on training, alternate runtimes, profiles, downloads, copying, and persistence.

- [ ] **Step 4: Verify only intended metadata changed and commit**

```powershell
git diff -- main.py distribution_metadata.py nuitka-package.config.yml requirements.txt models
git diff --check
git add tests/test_settings.py README.md AGENTS.md
git commit -m "docs: explain runtime AI model selection"
```

Expected: the first command prints no diff; the documentation and omission test are the only staged changes.

---

### Task 7: Run complete verification and hardware acceptance handoff

**Files:**
- Verify: `ai_model_selection.py`
- Verify: `ai_service.py`
- Verify: `ui.py`
- Verify: `tests/`
- Verify: repository metadata and bundled model

**Interfaces:**
- Consumes: all preceding task deliverables.
- Produces: a clean, tested source tree and an explicit list of hardware checks that remain external.

- [ ] **Step 1: Compile every source module including the new boundary**

```powershell
python -m py_compile main.py ui.py motion.py combined_motion.py ai_targeting.py ai_tracking.py ai_detection.py ai_model_selection.py ai_capture.py ai_zoom.py ai_service.py display_timing.py overlay.py makcu_service.py hotkeys.py settings.py sound_service.py liquid_widgets.py distribution_metadata.py
```

Expected: exit code 0 and no output.

- [ ] **Step 2: Run the complete hardware-free suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: every test passes, including stale switch, rollback-once, existing target tracking, combined movement, STOP, settings, packaging, and entry-point regressions.

- [ ] **Step 3: Verify runtime imports and the canonical bundled model**

```powershell
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
```

Expected: imports exit 0; self-check reports `ok: true`, the bundled model hash, and `DmlExecutionProvider`; metadata review reports the unchanged canonical default model/package plan.

- [ ] **Step 4: Inspect repository scope and commit any final test-only correction**

```powershell
git status --short
git diff --check
git diff --stat HEAD~6..HEAD
```

Expected: no generated build directories, external models, config files, backups, or logs are tracked. If verification required a focused correction, rerun its RED/GREEN test and commit only that correction with `fix: close AI model selection regression`.

- [ ] **Step 5: Report the connected-device acceptance checklist without running Nuitka**

Record these manual checks as not run unless a Makcu and compatible custom model are actually available:

```text
1. Switch Default -> compatible Custom while AI-only movement is active.
2. Switch during combined Jitter+AI and confirm no late movement from the old generation.
3. Verify Overlay-only Custom inference without Master or Makcu movement.
4. Reject an invalid-contract ONNX and observe automatic Default/previous rollback.
5. Press STOP during validation, candidate startup, and rollback; no auto-restart occurs.
6. Disconnect/reconnect and remove AI/Overlay demand during switching.
7. Run Trigger/Modifier gates, each source combination, Test 3s, hotkey, and shutdown.
8. Relaunch and confirm MODEL returns to Default without config/model copying.
```

Do not execute `gen.bat` or Nuitka for this feature handoff.
