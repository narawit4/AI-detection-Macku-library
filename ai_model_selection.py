"""Immutable AI model choices and one-shot background validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import logging
from pathlib import Path
import threading
from typing import Any

from ai_detection import ModelContractError, OnnxDetector, model_resource_path


LOGGER = logging.getLogger(__name__)


class ModelSelectionError(ValueError):
    """A concise model-selection failure safe to display in the UI."""


@dataclass(frozen=True)
class ModelChoice:
    path: Path
    display_name: str
    is_default: bool
    input_size: int | None = None


@dataclass(frozen=True)
class ModelValidationEvent:
    kind: str
    token: int
    choice: ModelChoice
    error_type: str | None = None
    safe_message: str | None = None


def bundled_model_choice() -> ModelChoice:
    path = model_resource_path().resolve()
    return ModelChoice(path, path.name, True, 320)


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
        with self._event_lock:
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
        except Exception:
            with self._event_lock:
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
            validated_choice = replace(choice, input_size=detector.input_size)
            event = ModelValidationEvent("ready", token, validated_choice)
        except Exception as error:
            LOGGER.exception("AI model validation failed for %s", choice.path)
            safe_message = (
                str(error)
                if isinstance(error, ModelContractError)
                else "AI model validation failed"
            )
            event = ModelValidationEvent(
                "error", token, choice, type(error).__name__, safe_message
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
        with self._event_lock:
            with self._lock:
                if self._active is not None:
                    self._active[1].set()
                    self._active = None

    def close(self) -> None:
        with self._event_lock:
            with self._lock:
                self._closed = True
                if self._active is not None:
                    self._active[1].set()
                    self._active = None
