"""Generation-safe capture and inference worker for AI aim mode."""

from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from .capture import (
    CENTER_320,
    FULL_DISPLAY,
    CapturedFrame,
    DxcamCapture,
    centered_region,
    validated_capture_mode,
)
from .detection import OnnxDetector, model_resource_path
from .targeting import (
    AimSettings,
    DetectionAnalysis,
    DetectionFrameSnapshot,
    TargetSnapshot,
    analyze_detections,
    validated_target_area,
)
from .zoom import (
    ZoomStabilityState,
    build_zoom_input,
    compose_zoom_refinement,
    limit_zoom_factor,
    observe_zoom_stability,
    record_zoom_refinement_miss,
    select_zoom_factor,
)


LOGGER = logging.getLogger(__name__)
_CAPTURE_GEOMETRY_ERROR = "AI captured frame geometry is inconsistent"


@dataclass(frozen=True)
class AiEvent:
    kind: str
    payload: Any = None
    targeting_revision: int | None = None


def _validate_captured_frame(
    captured: CapturedFrame,
    generation_mode: str,
) -> tuple[np.ndarray, int, int, int, int, int, int]:
    def fail() -> None:
        raise ValueError(_CAPTURE_GEOMETRY_ERROR)

    if not isinstance(captured, CapturedFrame):
        fail()
    try:
        captured_mode = validated_capture_mode(captured.mode)
    except ValueError:
        fail()
    if captured_mode != generation_mode:
        fail()
    frame = captured.pixels
    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[0] <= 0
        or frame.shape[1] <= 0
        or frame.shape[2] != 3
        or frame.dtype != np.uint8
    ):
        fail()
    frame_height, frame_width = frame.shape[:2]
    if (
        type(captured.output_width) is not int
        or captured.output_width <= 0
        or type(captured.output_height) is not int
        or captured.output_height <= 0
        or type(captured.capture_left) is not int
        or captured.capture_left < 0
        or type(captured.capture_top) is not int
        or captured.capture_top < 0
        or type(captured.capture_width) is not int
        or captured.capture_width <= 0
        or type(captured.capture_height) is not int
        or captured.capture_height <= 0
        or frame_width != captured.capture_width
        or frame_height != captured.capture_height
        or captured.capture_left + captured.capture_width > captured.output_width
        or captured.capture_top + captured.capture_height > captured.output_height
    ):
        fail()
    if generation_mode == CENTER_320:
        try:
            expected_region = centered_region(
                captured.output_width,
                captured.output_height,
            )
        except ValueError:
            fail()
        if (
            captured.capture_left,
            captured.capture_top,
            captured.capture_left + captured.capture_width,
            captured.capture_top + captured.capture_height,
        ) != expected_region:
            fail()
    elif (
        generation_mode == FULL_DISPLAY
        and (
            captured.capture_left != 0
            or captured.capture_top != 0
            or captured.capture_width != captured.output_width
            or captured.capture_height != captured.output_height
        )
    ):
        fail()
    return (
        frame,
        frame_width,
        frame_height,
        captured.output_width,
        captured.output_height,
        captured.capture_left,
        captured.capture_top,
    )


class AiService:
    """Own the AI model and capture resources on a daemon worker."""

    def __init__(
        self,
        event_sink: Callable[[AiEvent], None],
        model_path: Path | str | None = None,
        detector_factory: Callable[[Path | str], Any] = OnnxDetector,
        capture_factory: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        capture_fps: int = 120,
    ) -> None:
        self._event_sink = event_sink
        self._model_path = model_path
        self._detector_factory = detector_factory
        if type(capture_fps) is not int or capture_fps <= 0:
            capture_fps = 120
        self._capture_factory = (
            capture_factory
            if capture_factory is not None
            else lambda mode: DxcamCapture(mode=mode, target_fps=capture_fps)
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._event_lock = threading.RLock()
        self._generation = 0
        self._stop_event: threading.Event | None = None
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._closed = False
        self._latest: TargetSnapshot | None = None
        self._latest_detection: DetectionFrameSnapshot | None = None
        self._targeting_revision = 0
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

    @property
    def worker_active(self) -> bool:
        """Whether a worker still owns generation-local runtime resources."""
        with self._lock:
            worker = self._worker_thread
            if worker is None:
                return False
            if worker.is_alive():
                return True
            self._worker_thread = None
            return False

    def latest_snapshot(self) -> TargetSnapshot | None:
        with self._lock:
            return self._latest

    def latest_detection_snapshot(self) -> DetectionFrameSnapshot | None:
        with self._lock:
            return self._latest_detection

    def reset_targeting(self) -> int:
        """Immediately invalidate movement output without stopping inference."""
        with self._lock:
            self._targeting_revision += 1
            self._latest = None
            if self._latest_detection is not None:
                frame = self._latest_detection
                self._latest_detection = DetectionFrameSnapshot(
                    frame.sequence,
                    frame.captured_at,
                    frame.detections,
                    None,
                    frame.frame_width,
                    frame.frame_height,
                    frame.output_width,
                    frame.output_height,
                    frame.capture_left,
                    frame.capture_top,
                )
            return self._targeting_revision

    def _clear_snapshots_locked(self) -> None:
        self._latest = None
        self._latest_detection = None

    def start(
        self,
        settings_provider: Callable[[], AimSettings],
        zoom_gate_provider: Callable[[], bool] | None = None,
        *,
        model_path: Path | str | None = None,
        capture_mode: str = CENTER_320,
    ) -> int | None:
        generation_capture_mode = validated_capture_mode(capture_mode)
        if zoom_gate_provider is None:
            zoom_gate_provider = lambda: False
        with self._lock:
            if self._closed:
                return None
            if self._running:
                return self._generation
            retiring_worker = self._worker_thread
            if retiring_worker is not None:
                if retiring_worker.is_alive():
                    return None
                self._worker_thread = None
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
        worker: threading.Thread | None = None
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
                        model_path,
                        generation_capture_mode,
                    ),
                    name=f"AiInference-{generation}",
                    daemon=True,
                )
                self._worker_thread = worker
                worker.start()
            except Exception as error:
                stop_event.set()
                if worker is not None and self._worker_thread is worker:
                    self._worker_thread = None
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
        generation_model_path: Path | str | None,
        generation_capture_mode: str,
    ) -> None:
        capture = None
        try:
            if not self._is_current(generation, stop_event):
                return
            model_path = (
                generation_model_path
                if generation_model_path is not None
                else self._model_path or model_resource_path()
            )
            detector = self._detector_factory(model_path)
            if not self._is_current(generation, stop_event):
                return
            capture = self._capture_factory(generation_capture_mode)
            capture.start()
            provider = detector.provider
            with self._lock:
                if not self._is_current_locked(generation, stop_event):
                    return
                self._status = "ready"
                self._provider = provider
            self._emit_current(AiEvent("ready", provider), generation, stop_event)

            sequence = 0
            stability = ZoomStabilityState()
            active_target_area: str | None = None
            active_targeting_revision: int | None = None
            refinement_enabled = True
            published_factor = 1.0
            fps_started_at = self._clock()
            completed_inferences = 0
            while self._is_current(generation, stop_event):
                captured = capture.read()
                if captured is None:
                    stop_event.wait(0.001)
                    continue
                (
                    frame,
                    frame_width,
                    frame_height,
                    output_width,
                    output_height,
                    capture_left,
                    capture_top,
                ) = _validate_captured_frame(
                    captured,
                    generation_capture_mode,
                )
                captured_at = self._clock()
                sequence += 1
                with self._lock:
                    targeting_revision = self._targeting_revision
                settings = settings_provider()
                target_area = validated_target_area(settings.target_area)
                if (
                    target_area != active_target_area
                    or targeting_revision != active_targeting_revision
                ):
                    stability = ZoomStabilityState()
                    active_target_area = target_area
                    active_targeting_revision = targeting_revision
                if not self._is_current(generation, stop_event):
                    return
                base_detections = detector.detect(frame)
                if not self._is_current(generation, stop_event):
                    return
                base_analysis = analyze_detections(
                    base_detections,
                    settings,
                    sequence=sequence,
                    captured_at=captured_at,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    output_width=output_width,
                    output_height=output_height,
                    capture_left=capture_left,
                    capture_top=capture_top,
                )
                factor = 1.0
                published = base_analysis
                gate_active = bool(zoom_gate_provider())
                if not gate_active:
                    stability = ZoomStabilityState()
                else:
                    stability = observe_zoom_stability(
                        stability,
                        base_analysis.target,
                        captured_at,
                    )
                if refinement_enabled and gate_active:
                    try:
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
                    except Exception:
                        LOGGER.exception(
                            "Adaptive AI zoom disabled for generation %s",
                            generation,
                        )
                        refinement_enabled = False
                        factor = 1.0
                        published = base_analysis
                with self._lock:
                    if not self._is_current_locked(generation, stop_event):
                        return
                    if targeting_revision != self._targeting_revision:
                        factor = 1.0
                        self._latest = None
                        self._latest_detection = DetectionFrameSnapshot(
                            published.frame.sequence,
                            published.frame.captured_at,
                            published.frame.detections,
                            None,
                            published.frame.frame_width,
                            published.frame.frame_height,
                            published.frame.output_width,
                            published.frame.output_height,
                            published.frame.capture_left,
                            published.frame.capture_top,
                        )
                    else:
                        self._latest = published.target
                        self._latest_detection = published.frame
                if factor != published_factor:
                    self._emit_current(
                        AiEvent("zoom", factor, targeting_revision),
                        generation,
                        stop_event,
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
