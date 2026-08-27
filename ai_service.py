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
from ai_targeting import (
    AimSettings,
    DetectionAnalysis,
    DetectionFrameSnapshot,
    TargetSnapshot,
)
from ai_tracking import TrackerState, observe_detections
from ai_zoom import (
    ZoomStabilityState,
    build_zoom_input,
    compose_zoom_refinement,
    limit_zoom_factor,
    movement_is_confirmed,
    observe_zoom_stability,
    record_zoom_refinement_miss,
    select_zoom_factor,
)


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
        self._latest_detection: DetectionFrameSnapshot | None = None
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

    def latest_detection_snapshot(self) -> DetectionFrameSnapshot | None:
        with self._lock:
            return self._latest_detection

    def _clear_snapshots_locked(self) -> None:
        self._latest = None
        self._latest_detection = None

    def start(
        self,
        settings_provider: Callable[[], AimSettings],
        zoom_gate_provider: Callable[[], bool] | None = None,
    ) -> int | None:
        if zoom_gate_provider is None:
            zoom_gate_provider = lambda: False
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
            self._clear_snapshots_locked()
            self._status = "loading"
            self._provider = None

        self._emit_current(AiEvent("loading"), generation, stop_event)
        start_error = None
        failure_generation = None
        with self._lock:
            if not self._is_current_locked(generation, stop_event):
                return generation
            try:
                worker = threading.Thread(
                    target=self._worker,
                    args=(
                        generation,
                        stop_event,
                        settings_provider,
                        zoom_gate_provider,
                    ),
                    name=f"AiInference-{generation}",
                    daemon=True,
                )
                worker.start()
            except Exception as error:
                stop_event.set()
                self._generation += 1
                failure_generation = self._generation
                self._running = False
                self._clear_snapshots_locked()
                self._status = "error"
                self._provider = None
                self._stop_event = None
                start_error = error
        if start_error is not None:
            LOGGER.error(
                "AI inference worker could not start",
                exc_info=(
                    type(start_error),
                    start_error,
                    start_error.__traceback__,
                ),
            )
            self._emit_status_current(
                AiEvent(
                    "error",
                    f"{type(start_error).__name__}: AI service failed",
                ),
                failure_generation,
                "error",
            )
            return None
        return generation

    def stop(self, reason: str = "manual") -> None:
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()
        stopped_generation = None
        with self._lock:
            current_stop_event = self._stop_event
            if current_stop_event is not None:
                current_stop_event.set()
            if not self._closed and (
                self._running or self._status != "stopped"
            ):
                self._generation += 1
                stopped_generation = self._generation
                self._running = False
                self._clear_snapshots_locked()
                self._status = "stopped"
                self._provider = None
                self._stop_event = None
        if stopped_generation is not None:
            self._emit_stopped_current(
                AiEvent("stopped", reason), stopped_generation
            )
        else:
            self._cross_event_barrier()

    def close(self) -> None:
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()
        with self._lock:
            current_stop_event = self._stop_event
            if current_stop_event is not None:
                current_stop_event.set()
            if not self._closed:
                self._closed = True
                self._generation += 1
                self._running = False
                self._clear_snapshots_locked()
                self._status = "stopped"
                self._provider = None
                self._stop_event = None
        self._cross_event_barrier()

    def _cross_event_barrier(self) -> None:
        with self._event_lock:
            pass

    def _is_current(self, generation: int, stop_event: threading.Event) -> bool:
        if stop_event.is_set():
            return False
        with self._lock:
            return self._is_current_locked(generation, stop_event)

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

    def _emit_status_current(
        self,
        event: AiEvent,
        generation: int,
        status: str,
    ) -> None:
        with self._event_lock:
            with self._lock:
                current = (
                    not self._closed
                    and not self._running
                    and self._generation == generation
                    and self._status == status
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
        zoom_gate_provider: Callable[[], bool],
    ) -> None:
        capture = None
        try:
            if not self._is_current(generation, stop_event):
                return
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
            tracker_state = TrackerState()
            stability = ZoomStabilityState()
            refinement_enabled = True
            published_factor = 1.0
            fps_started_at = self._clock()
            completed_inferences = 0
            while self._is_current(generation, stop_event):
                frame = capture.read()
                if frame is None:
                    stop_event.wait(0.001)
                    continue
                captured_at = self._clock()
                sequence += 1
                settings = settings_provider()
                if not self._is_current(generation, stop_event):
                    return
                base_detections = detector.detect(frame)
                if not self._is_current(generation, stop_event):
                    return
                tracked = observe_detections(
                    tracker_state,
                    base_detections,
                    settings,
                    sequence=sequence,
                    captured_at=captured_at,
                )
                tracker_state = tracked.state
                base_analysis = tracked.analysis
                factor = 1.0
                published = base_analysis
                gate_active = False
                if refinement_enabled:
                    try:
                        gate_active = bool(zoom_gate_provider())
                        if not gate_active:
                            stability = ZoomStabilityState()
                        else:
                            stability = observe_zoom_stability(
                                stability,
                                base_analysis.target,
                                captured_at,
                            )
                            selected = base_analysis.frame.selected_index
                            if selected is not None and base_analysis.target is not None:
                                seed = base_analysis.frame.detections[selected]
                                requested_factor = select_zoom_factor(
                                    seed,
                                    base_analysis.target,
                                )
                                applied_factor = limit_zoom_factor(
                                    requested_factor,
                                    stability,
                                    captured_at,
                                )
                                if applied_factor > 1.0:
                                    if not bool(zoom_gate_provider()):
                                        gate_active = False
                                        stability = ZoomStabilityState()
                                    else:
                                        if not self._is_current(
                                            generation, stop_event
                                        ):
                                            return
                                        zoomed, transform = build_zoom_input(
                                            frame,
                                            base_analysis.target,
                                            applied_factor,
                                        )
                                        if not self._is_current(
                                            generation, stop_event
                                        ):
                                            return
                                        refined_detections = detector.detect(zoomed)
                                        if not self._is_current(
                                            generation, stop_event
                                        ):
                                            return
                                        if bool(zoom_gate_provider()):
                                            refined = compose_zoom_refinement(
                                                base_analysis,
                                                refined_detections,
                                                transform,
                                                settings,
                                            )
                                            if refined is None:
                                                stability = record_zoom_refinement_miss(
                                                    stability,
                                                    self._clock(),
                                                )
                                            else:
                                                published = refined
                                                factor = applied_factor
                                        else:
                                            gate_active = False
                                            stability = ZoomStabilityState()
                            if gate_active and not movement_is_confirmed(stability):
                                published = DetectionAnalysis(None, published.frame)
                    except Exception:
                        LOGGER.exception(
                            "Adaptive AI zoom disabled for generation %s",
                            generation,
                        )
                        refinement_enabled = False
                        stability = ZoomStabilityState()
                        factor = 1.0
                        published = base_analysis
                with self._lock:
                    if not self._is_current_locked(generation, stop_event):
                        return
                    self._latest = published.target
                    self._latest_detection = published.frame
                if factor != published_factor:
                    self._emit_current(
                        AiEvent("zoom", factor), generation, stop_event
                    )
                    published_factor = factor
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
            self._clear_snapshots_locked()
            self._status = "error"
            self._provider = None
        self._emit_current(
            AiEvent("error", f"{type(error).__name__}: AI service failed"),
            generation,
            stop_event,
        )
