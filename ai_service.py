"""Generation-safe capture and inference worker for AI aim mode."""

from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Any

from ai_capture import DxcamCapture
from ai_detection import OnnxDetector, model_resource_path
from ai_targeting import AimSettings, TargetSnapshot, select_target


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AiEvent:
    kind: str
    payload: Any = None


class AiService:
    """Own the AI model and capture resources on a daemon worker."""

    def __init__(
        self,
        event_sink: Callable[[AiEvent], None],
        model_path: Path | str | None = None,
        detector_factory: Callable[[Path | str], Any] = OnnxDetector,
        capture_factory: Callable[[], Any] = DxcamCapture,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._event_sink = event_sink
        self._model_path = model_path
        self._detector_factory = detector_factory
        self._capture_factory = capture_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._event_lock = threading.RLock()
        self._generation = 0
        self._stop_event: threading.Event | None = None
        self._running = False
        self._closed = False
        self._latest: TargetSnapshot | None = None
        self._status = "stopped"
        self._provider: str | None = None

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def provider(self) -> str | None:
        with self._lock:
            return self._provider

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def latest_snapshot(self) -> TargetSnapshot | None:
        with self._lock:
            return self._latest

    def start(self, settings_provider: Callable[[], AimSettings]) -> int | None:
        with self._lock:
            if self._closed:
                return None
            if self._running:
                return self._generation
            self._generation += 1
            generation = self._generation
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._running = True
            self._latest = None
            self._status = "loading"
            self._provider = None

        self._emit_current(AiEvent("loading"), generation, stop_event)
        threading.Thread(
            target=self._worker,
            args=(generation, stop_event, settings_provider),
            name=f"AiInference-{generation}",
            daemon=True,
        ).start()
        return generation

    def stop(self, reason: str = "manual") -> None:
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()
        with self._lock:
            if self._closed or (not self._running and self._status == "stopped"):
                return
            self._generation += 1
            stopped_generation = self._generation
            self._running = False
            self._latest = None
            self._status = "stopped"
            self._provider = None
            self._stop_event = None
        self._emit_stopped_current(
            AiEvent("stopped", reason), stopped_generation
        )

    def close(self) -> None:
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._running = False
            self._latest = None
            self._status = "stopped"
            self._provider = None
            self._stop_event = None
        with self._event_lock:
            pass

    def _is_current(self, generation: int, stop_event: threading.Event) -> bool:
        if stop_event.is_set():
            return False
        with self._lock:
            return (
                not self._closed
                and self._running
                and self._generation == generation
                and self._stop_event is stop_event
            )

    def _emit_current(
        self,
        event: AiEvent,
        generation: int,
        stop_event: threading.Event,
    ) -> None:
        with self._event_lock:
            if self._is_current(generation, stop_event):
                self._emit(event)

    def _emit_stopped_current(self, event: AiEvent, generation: int) -> None:
        with self._event_lock:
            with self._lock:
                current = (
                    not self._closed
                    and not self._running
                    and self._generation == generation
                    and self._status == "stopped"
                )
            if current:
                self._emit(event)

    def _emit(self, event: AiEvent) -> None:
        try:
            self._event_sink(event)
        except Exception:
            LOGGER.exception("AI event sink failed for %s", event.kind)

    def _worker(
        self,
        generation: int,
        stop_event: threading.Event,
        settings_provider: Callable[[], AimSettings],
    ) -> None:
        capture = None
        try:
            model_path = self._model_path or model_resource_path()
            detector = self._detector_factory(model_path)
            if not self._is_current(generation, stop_event):
                return
            capture = self._capture_factory()
            capture.start()
            provider = detector.provider
            with self._lock:
                if not self._is_current_locked(generation, stop_event):
                    return
                self._status = "ready"
                self._provider = provider
            self._emit_current(AiEvent("ready", provider), generation, stop_event)

            sequence = 0
            previous = None
            fps_started_at = self._clock()
            completed_inferences = 0
            while self._is_current(generation, stop_event):
                frame = capture.read()
                if frame is None:
                    stop_event.wait(0.001)
                    continue
                captured_at = self._clock()
                detections = detector.detect(frame)
                sequence += 1
                selected = select_target(
                    detections,
                    settings_provider(),
                    sequence=sequence,
                    captured_at=captured_at,
                    previous=previous,
                )
                if not self._is_current(generation, stop_event):
                    return
                with self._lock:
                    if not self._is_current_locked(generation, stop_event):
                        return
                    self._latest = selected
                previous = selected
                completed_inferences += 1
                now = self._clock()
                elapsed = now - fps_started_at
                if elapsed >= 1.0:
                    self._emit_current(
                        AiEvent("fps", completed_inferences / elapsed),
                        generation,
                        stop_event,
                    )
                    fps_started_at = now
                    completed_inferences = 0
        except Exception as error:
            LOGGER.exception("AI inference worker failed")
            self._fail_current(error, generation, stop_event)
        finally:
            if capture is not None:
                try:
                    capture.close()
                except Exception:
                    LOGGER.exception("AI capture cleanup failed")
            with self._lock:
                if self._generation == generation and self._stop_event is stop_event:
                    self._running = False
                    self._stop_event = None

    def _is_current_locked(
        self,
        generation: int,
        stop_event: threading.Event,
    ) -> bool:
        return (
            not self._closed
            and self._running
            and self._generation == generation
            and self._stop_event is stop_event
            and not stop_event.is_set()
        )

    def _fail_current(
        self,
        error: Exception,
        generation: int,
        stop_event: threading.Event,
    ) -> None:
        with self._lock:
            if not self._is_current_locked(generation, stop_event):
                return
            self._latest = None
            self._status = "error"
            self._provider = None
        self._emit_current(
            AiEvent("error", f"{type(error).__name__}: AI service failed"),
            generation,
            stop_event,
        )
