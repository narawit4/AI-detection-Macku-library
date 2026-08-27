"""Generation-safe Makcu connection lifecycle for the Jitter UI.

The service deliberately has no Tkinter dependency.  All callbacks become
small :class:`ServiceEvent` values and are delivered to the caller's sink;
the application layer is responsible for marshalling those values to Tk's
main thread.
"""

from dataclasses import dataclass, field
import logging
import math
import threading
import time
from typing import Any, Callable

from makcu import MouseButton, create_controller
from ai_targeting import AimMovementEngine, AimSettings, TargetSnapshot
from combined_motion import CombinedMotionEngine, MotionSources
from motion import MotionSettings, PairedPulseEngine


_MISSING = object()
_CONNECTION_FAILURE_MESSAGE = "Makcu unavailable; check USB and reconnect"
LOGGER = logging.getLogger(__name__)


BUTTON_NAMES = {
    MouseButton.LEFT: "Left",
    MouseButton.RIGHT: "Right",
    MouseButton.MIDDLE: "Middle",
    MouseButton.MOUSE4: "Mouse4",
    MouseButton.MOUSE5: "Mouse5",
}


@dataclass(frozen=True)
class ServiceEvent:
    kind: str
    payload: Any = None
    # Motion terminal/error events carry the generation reserved for their
    # originating worker.  ``compare=False`` keeps equality compatible with
    # existing consumers that compare only the historical kind/payload pair.
    motion_generation: int | None = field(default=None, compare=False)


class MakcuService:
    """Own one generation of a Makcu controller at a time.

    Connection creation, stale-controller cleanup, and explicit disconnects
    run on daemon threads when initiated by the public lifecycle methods.
    The service never knows about Tkinter and never updates UI state itself.
    """

    def __init__(
        self,
        event_sink: Callable[[ServiceEvent], None],
        controller_factory: Callable[..., Any] = create_controller,
        engine_factory: Callable[[], Any] = PairedPulseEngine,
        aim_engine_factory: Callable[[], Any] | None = None,
        combined_engine_factory: Callable[[MotionSources], Any] | None = None,
        ai_poll_hz: float = 240.0,
    ) -> None:
        self._event_sink = event_sink
        self._controller_factory = controller_factory
        self._engine_factory = engine_factory
        self._aim_engine_factory = aim_engine_factory
        try:
            validated_ai_poll_hz = (
                float(ai_poll_hz) if not isinstance(ai_poll_hz, bool) else 240.0
            )
        except (TypeError, ValueError, OverflowError):
            validated_ai_poll_hz = 240.0
        if not math.isfinite(validated_ai_poll_hz) or validated_ai_poll_hz <= 0.0:
            validated_ai_poll_hz = 240.0
        self._ai_poll_hz = validated_ai_poll_hz
        self._combined_engine_factory = (
            combined_engine_factory
            if combined_engine_factory is not None
            else lambda sources: CombinedMotionEngine(
                sources,
                jitter_engine_factory=engine_factory,
                aim_engine_factory=aim_engine_factory,
                ai_poll_hz=self._ai_poll_hz,
            )
        )
        self._lock = threading.RLock()
        self._generation = 0
        self._controller: Any | None = None
        self._connected = False
        self._closed = False
        self._connection_transition = 0
        self._connection_thread: threading.Thread | None = None
        self._connection_request_generation: int | None = None
        self._retained_controllers: list[Any] = []
        self._setup_disconnected: set[int] = set()
        self._disconnect_notified: set[int] = set()
        self._move_barrier = threading.Lock()
        self._motion_cancel_lock = threading.Lock()
        self._motion_dispatch_done = threading.Event()
        self._motion_dispatch_done.set()
        self._motion_dispatch_owner: threading.Thread | None = None
        self._motion_dispatch_queue: list[
            tuple[ServiceEvent, Callable[[], bool] | None]
        ] = []
        self._motion_start_cancel_epoch = 0
        self._motion_generation = 0
        self._motion_stop = threading.Event()
        self._motion_thread: threading.Thread | None = None
        self._health_thread: threading.Thread | None = None
        self._health_stop = threading.Event()
        self._health_stop.set()
        self._motion_active = False
        self._motion_stop_reasons: dict[int, str | None] = {}

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def controller(self) -> Any | None:
        with self._lock:
            return self._controller

    @property
    def connection_generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def motion_active(self) -> bool:
        with self._lock:
            return self._motion_active

    def _emit(self, event: ServiceEvent) -> None:
        # A UI event sink normally just queues work for Tk's thread.  Keep a
        # bad sink from killing a daemon connection worker or cleanup thread.
        try:
            self._event_sink(event)
        except Exception:
            pass

    def _begin_connection_locked(self) -> tuple[int, bool]:
        self._health_stop.set()
        self._health_stop = threading.Event()
        self._generation += 1
        generation = self._generation
        self._controller = None
        self._connected = False
        self._setup_disconnected.discard(generation)
        self._disconnect_notified.discard(generation)
        should_dispatch = self._reserve_event_locked(
            ServiceEvent("connecting"),
            lambda: generation == self._generation and not self._closed,
        )
        return generation, should_dispatch

    def _begin_connection(self) -> int:
        with self._lock:
            generation, should_dispatch = self._begin_connection_locked()
        if should_dispatch:
            self._drain_event_dispatch()
        return generation

    def _reserve_event_locked(
        self,
        event: ServiceEvent,
        current_guard: Callable[[], bool] | None = None,
    ) -> bool:
        # Dispatch has two states, both changed only under ``_lock``: idle
        # (no owner, done set) and draining (one owner, done clear).  Producers
        # only append here; the owner invokes the external sink without any
        # service lock and then returns to ``_lock`` for the next reservation.
        self._motion_dispatch_queue.append((event, current_guard))
        if self._motion_dispatch_owner is not None:
            return False
        self._motion_dispatch_owner = threading.current_thread()
        self._motion_dispatch_done.clear()
        return True

    def _drain_event_dispatch(self) -> None:
        while True:
            with self._lock:
                if not self._motion_dispatch_queue:
                    self._motion_dispatch_owner = None
                    self._motion_dispatch_done.set()
                    return
                event, current_guard = self._motion_dispatch_queue.pop(0)
                if current_guard is not None and not current_guard():
                    continue
            self._emit(event)

    def _wait_for_motion_dispatch(self, timeout: float | None = None) -> bool:
        with self._lock:
            if threading.current_thread() is self._motion_dispatch_owner:
                return True
            done = self._motion_dispatch_done
        return done.wait(timeout)

    def connect(self) -> int | None:
        """Start a fresh connection worker and return its generation."""
        with self._lock:
            if self._closed:
                return None
        return self.reconnect()

    def _retain_controller_locked(self, controller: Any | None) -> None:
        if controller is None:
            return
        if all(controller is not retained for retained in self._retained_controllers):
            self._retained_controllers.append(controller)

    def _retain_controller(self, controller: Any) -> None:
        with self._lock:
            self._retain_controller_locked(controller)

    def _reserve_connection_worker_locked(self) -> threading.Thread | None:
        if self._connection_thread is not None:
            return None
        thread = threading.Thread(
            target=self._connection_lifecycle_worker,
            name=f"MakcuConnect-{self._generation}",
            daemon=True,
        )
        self._connection_thread = thread
        return thread

    def _retire_connection_worker_locked(self) -> None:
        if self._connection_thread is threading.current_thread():
            self._connection_thread = None

    def _mark_connection_unavailable_locked(
        self,
        generation: int,
    ) -> bool:
        if generation != self._generation or self._closed:
            return False
        self._controller = None
        self._connected = False
        already_notified = generation in self._disconnect_notified
        self._disconnect_notified.add(generation)
        if already_notified:
            return False
        return self._reserve_event_locked(
            ServiceEvent("disconnected", _CONNECTION_FAILURE_MESSAGE),
            lambda: (
                generation == self._generation
                and not self._closed
                and self._controller is None
                and not self._connected
            ),
        )

    def _launch_connection_worker(self, thread: threading.Thread | None) -> None:
        if thread is None:
            return
        try:
            thread.start()
        except Exception:
            LOGGER.exception("Makcu connection worker could not start")
            should_dispatch = False
            with self._lock:
                if self._connection_thread is thread:
                    generation = self._connection_request_generation
                    self._connection_request_generation = None
                    if generation is not None:
                        should_dispatch = self._mark_connection_unavailable_locked(
                            generation
                        )
                    self._connection_thread = None
            if should_dispatch:
                self._drain_event_dispatch()

    def _connection_lifecycle_worker(self) -> None:
        """Serialize movement, teardown, and the latest requested factory."""
        try:
            while True:
                # Reconnect invalidates state and signals cancellation before
                # this worker starts.  Crossing the same barrier as move()
                # proves that no call still owns the retiring controller.
                with self._move_barrier:
                    pass

                with self._lock:
                    retained = (
                        self._retained_controllers[0]
                        if self._retained_controllers
                        else None
                    )
                    generation = self._connection_request_generation
                    closed = self._closed

                if retained is not None:
                    if not self._disconnect_controller(retained):
                        should_dispatch = False
                        with self._lock:
                            generation = self._connection_request_generation
                            self._connection_request_generation = None
                            if generation is not None:
                                should_dispatch = (
                                    self._mark_connection_unavailable_locked(
                                        generation
                                    )
                                )
                            self._retire_connection_worker_locked()
                        if should_dispatch:
                            self._drain_event_dispatch()
                        return
                    with self._lock:
                        self._retained_controllers = [
                            controller
                            for controller in self._retained_controllers
                            if controller is not retained
                        ]
                    continue

                if closed or generation is None:
                    with self._lock:
                        # A reconnect that won the lock first leaves a new
                        # request here, so loop instead of dropping it while
                        # this sole lifecycle worker retires.
                        if (
                            not self._closed
                            and self._connection_request_generation is not None
                        ):
                            continue
                        self._retire_connection_worker_locked()
                    return

                cleanup_proven = self._connect_worker(generation)
                should_dispatch = False
                with self._lock:
                    current_request = self._connection_request_generation
                    if not cleanup_proven:
                        self._connection_request_generation = None
                        if current_request is not None:
                            should_dispatch = (
                                self._mark_connection_unavailable_locked(
                                    current_request
                                )
                            )
                        self._retire_connection_worker_locked()
                        should_return = True
                    elif self._retained_controllers:
                        should_return = False
                    elif current_request != generation:
                        should_return = False
                    else:
                        self._connection_request_generation = None
                        self._retire_connection_worker_locked()
                        should_return = True
                if should_dispatch:
                    self._drain_event_dispatch()
                if should_return:
                    return
        except Exception:
            LOGGER.exception("Makcu connection lifecycle worker failed")
            should_dispatch = False
            with self._lock:
                generation = self._connection_request_generation
                self._connection_request_generation = None
                if generation is not None:
                    should_dispatch = self._mark_connection_unavailable_locked(
                        generation
                    )
                self._retire_connection_worker_locked()
            if should_dispatch:
                self._drain_event_dispatch()

    def join_connection(self, timeout: float | None = None) -> None:
        """Bounded lifecycle/testing helper; do not call this from Tk."""
        with self._lock:
            thread = self._connection_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _start_health_worker(self, generation: int, controller: Any) -> None:
        with self._lock:
            if (
                self._closed
                or generation != self._generation
                or controller is not self._controller
            ):
                return
            stop_event = self._health_stop
            thread = threading.Thread(
                target=self._health_worker,
                args=(generation, controller, stop_event),
                name=f"MakcuHealth-{generation}",
                daemon=True,
            )
            self._health_thread = thread
        try:
            thread.start()
        except Exception:
            with self._lock:
                if self._health_thread is thread:
                    self._health_thread = None
            raise

    def _health_worker(
        self,
        generation: int,
        controller: Any,
        stop_event: threading.Event,
    ) -> None:
        health_error_active = False
        try:
            while not stop_event.wait(0.1):
                with self._lock:
                    if (
                        self._closed
                        or generation != self._generation
                        or controller is not self._controller
                        or stop_event is not self._health_stop
                    ):
                        return
                try:
                    connected = bool(controller.is_connected())
                except Exception:
                    if not health_error_active:
                        LOGGER.exception("Makcu connection health check failed")
                    health_error_active = True
                    connected = False
                else:
                    health_error_active = False
                self._connection_changed(generation, connected)
        finally:
            with self._lock:
                if self._health_thread is threading.current_thread():
                    self._health_thread = None

    @staticmethod
    def _disconnect_controller(controller: Any) -> bool:
        try:
            controller.disconnect()
        except Exception:
            LOGGER.exception("Makcu controller disconnect failed")
            return False
        return True

    def start_motion(
        self,
        settings_provider: Callable[[], Any],
        duration_s: float | None = None,
    ) -> bool:
        """Start one interruptible movement worker for the active controller."""
        return self.start_motion_source(settings_provider, duration_s) is not None

    def start_motion_source(
        self,
        settings_provider: Callable[[], MotionSettings],
        duration_s: float | None = None,
    ) -> int | None:
        """Start motion and return its reserved source generation, if accepted."""
        return self.start_composite_motion_source(
            MotionSources(True, False),
            settings_provider,
            lambda: None,
            AimSettings,
            duration_s,
        )

    def start_ai_motion(
        self,
        snapshot_provider: Callable[[], TargetSnapshot | None],
        settings_provider: Callable[[], AimSettings],
        duration_s: float | None = None,
    ) -> bool:
        """Start AI target movement through the active controller."""
        return (
            self.start_ai_motion_source(
                snapshot_provider,
                settings_provider,
                duration_s,
            )
            is not None
        )

    def start_ai_motion_source(
        self,
        snapshot_provider: Callable[[], TargetSnapshot | None],
        settings_provider: Callable[[], AimSettings],
        duration_s: float | None = None,
    ) -> int | None:
        """Start AI motion and return its source generation, if accepted."""
        return self.start_composite_motion_source(
            MotionSources(False, True),
            MotionSettings,
            snapshot_provider,
            settings_provider,
            duration_s,
        )

    def start_composite_motion_source(
        self,
        sources: MotionSources,
        motion_settings_provider: Callable[[], MotionSettings],
        target_provider: Callable[[], TargetSnapshot | None],
        aim_settings_provider: Callable[[], AimSettings],
        duration_s: float | None = None,
    ) -> int | None:
        """Start selected motion sources through one controller worker."""
        if not isinstance(sources, MotionSources) or not sources.any:
            return None
        return self._start_motion_job(
            sources,
            motion_settings_provider,
            target_provider,
            aim_settings_provider,
            duration_s,
        )

    def _start_motion_job(
        self,
        sources: MotionSources,
        motion_settings_provider: Callable[[], MotionSettings],
        target_provider: Callable[[], TargetSnapshot | None],
        aim_settings_provider: Callable[[], AimSettings],
        duration_s: float | None,
    ) -> int | None:
        start_cancel_epoch: int | None = None
        while True:
            with self._lock:
                if self._closed or not self._connected or self._controller is None:
                    return None
                if (
                    start_cancel_epoch is not None
                    and start_cancel_epoch != self._motion_start_cancel_epoch
                ):
                    return None
                if self._motion_active:
                    return self._motion_generation
                dispatch_owner = self._motion_dispatch_owner
                if (
                    dispatch_owner is not None
                    and dispatch_owner is not threading.current_thread()
                ):
                    if start_cancel_epoch is None:
                        start_cancel_epoch = self._motion_start_cancel_epoch
                    dispatch_done = self._motion_dispatch_done
                else:
                    # The owner is already inside the prior callback, so a
                    # direct reentrant start has observed that terminal event.
                    # Every other thread waits outside ``_lock`` and retries
                    # this complete state check before allocating a generation.
                    try:
                        duration = (
                            None if duration_s is None else float(duration_s)
                        )
                        if duration is not None and not math.isfinite(duration):
                            return None
                        if duration is not None:
                            duration = max(0.0, duration)
                    except (TypeError, ValueError, OverflowError):
                        return None
                    connection_generation = self._generation
                    expected_controller = self._controller
                    stop_event = threading.Event()
                    with self._motion_cancel_lock:
                        self._motion_generation += 1
                        motion_generation = self._motion_generation
                        self._motion_stop = stop_event
                        self._motion_stop_reasons[motion_generation] = None
                    self._motion_active = True
                    thread = threading.Thread(
                        target=self._motion_worker,
                        args=(
                            motion_generation,
                            connection_generation,
                            stop_event,
                            sources,
                            motion_settings_provider,
                            target_provider,
                            aim_settings_provider,
                            duration,
                            expected_controller,
                        ),
                        name=f"MakcuMotion-{motion_generation}",
                        daemon=True,
                    )
                    self._motion_thread = thread
                    break
            dispatch_done.wait()
        try:
            thread.start()
        except Exception:
            with self._lock:
                if self._motion_generation == motion_generation:
                    self._motion_active = False
                    self._motion_thread = None
                    with self._motion_cancel_lock:
                        self._motion_stop_reasons.pop(motion_generation, None)
            return None
        return motion_generation

    def stop_motion(self, reason: str = "manual") -> None:
        """Signal cancellation immediately, then serialize the stop return."""
        reason = str(reason or "manual")

        self._signal_motion_cancellation(reason)

        # Crossing this barrier guarantees that an in-flight move has
        # returned.  The state lock remains available to reconnect/close while
        # a device call is hung, which keeps Tk lifecycle actions nonblocking.
        with self._move_barrier:
            with self._lock:
                self._invalidate_motion_locked(reason)

    def cancel_motion(self, reason: str = "manual") -> None:
        """Cancel uncommitted reports without waiting for device I/O.

        A report that already passed the final check under ``_move_barrier``
        is in flight and may finish after this method returns.  The stop event
        prevents the worker from committing a subsequent report.
        """
        reason = str(reason or "manual")
        self._signal_motion_cancellation(reason)
        with self._lock:
            self._invalidate_motion_locked(reason)

    def _signal_motion_cancellation(self, reason: str) -> None:
        """Wake the current motion generation without waiting for move()."""
        reason = str(reason or "manual")

        # This first signal deliberately takes no lock.  A controller move can
        # take an arbitrary duration, but lifecycle actions must wake every
        # waiter immediately.  Revalidation handles an event that became stale
        # because a new generation started concurrently.
        self._motion_stop.set()
        with self._motion_cancel_lock:
            generation = self._motion_generation
            self._motion_stop.set()
            if (
                generation in self._motion_stop_reasons
                and self._motion_stop_reasons[generation] is None
            ):
                self._motion_stop_reasons[generation] = reason

    def _invalidate_motion_locked(self, reason: str) -> None:
        """Invalidate starts and re-signal the generation under state lock."""
        # Python integers are unbounded, so this monotonic epoch cannot wrap
        # around to make an obsolete start waiter current again.
        self._motion_start_cancel_epoch += 1
        if not self._motion_active:
            return
        with self._motion_cancel_lock:
            generation = self._motion_generation
            self._motion_stop.set()
            if self._motion_stop_reasons.get(generation) is None:
                self._motion_stop_reasons[generation] = reason

    def join_motion(self, timeout: float | None = None) -> None:
        """Bounded lifecycle/testing helper; do not call this from the Tk thread."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._lock:
            thread = self._motion_thread
        if thread is not None and thread is not threading.current_thread():
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            thread.join(remaining)
        remaining = (
            None if deadline is None else max(0.0, deadline - time.monotonic())
        )
        self._wait_for_motion_dispatch(remaining)

    def _motion_worker(
        self,
        motion_generation: int,
        connection_generation: int,
        stop_event: threading.Event,
        sources: MotionSources,
        motion_settings_provider: Callable[[], MotionSettings],
        target_provider: Callable[[], TargetSnapshot | None],
        aim_settings_provider: Callable[[], AimSettings],
        duration_s: float | None,
        expected_controller: Any | None = None,
    ) -> None:
        engine: Any | None = None
        reason: str | None = None
        error_payload: str | None = None
        started = time.perf_counter()
        previous_tick = started
        if expected_controller is None:
            with self._lock:
                expected_controller = self._controller
        try:
            try:
                engine = self._combined_engine_factory(sources)
            except Exception as exc:
                error_payload = f"{type(exc).__name__}: {exc}"
            while error_payload is None:
                now = time.perf_counter()
                elapsed = max(0.0, now - started)
                if duration_s is not None and elapsed >= duration_s:
                    reason = "duration_complete"
                    break
                with self._lock:
                    if stop_event.is_set():
                        with self._motion_cancel_lock:
                            reason = (
                                self._motion_stop_reasons.get(motion_generation)
                                or "manual"
                            )
                        break
                    if (
                        motion_generation != self._motion_generation
                        or connection_generation != self._generation
                        or self._closed
                        or not self._connected
                        or self._controller is None
                    ):
                        reason = "disconnected"
                        break
                    controller = self._controller
                try:
                    motion_settings = motion_settings_provider()
                    tick_started = time.perf_counter()
                    dt = max(0.0, min(tick_started - previous_tick, 0.1))
                    previous_tick = tick_started
                    report_x, report_y = engine.step(
                        motion_settings,
                        target_provider(),
                        aim_settings_provider(),
                        dt=dt,
                        elapsed=elapsed,
                        now=tick_started,
                    )
                    interval = engine.poll_interval(motion_settings)
                except Exception as exc:
                    error_payload = f"{type(exc).__name__}: {exc}"
                    break
                if report_x or report_y:
                    # Serialize the stop check with the move call: once
                    # stop_motion() returns, this worker cannot send another
                    # report for this generation.
                    with self._move_barrier:
                        with self._lock:
                            if (
                                duration_s is not None
                                and max(0.0, time.perf_counter() - started)
                                >= duration_s
                            ):
                                reason = "duration_complete"
                                break
                            if (
                                stop_event.is_set()
                                or motion_generation != self._motion_generation
                                or connection_generation != self._generation
                                or self._closed
                                or not self._connected
                                or controller is not self._controller
                            ):
                                with self._motion_cancel_lock:
                                    reason = (
                                        self._motion_stop_reasons.get(
                                            motion_generation
                                        )
                                        or "disconnected"
                                    )
                                break
                        try:
                            # Passing the guarded state check commits this one
                            # report as in flight.  Nonblocking cancellation may
                            # return while it finishes, but cannot commit the
                            # worker's next report.
                            controller.move(report_x, report_y)
                        except Exception as exc:
                            error_payload = f"{type(exc).__name__}: {exc}"
                            break
                stop_event.wait(max(0.0, interval - (time.perf_counter() - tick_started)))
        finally:
            terminal_event: ServiceEvent | None = None
            should_dispatch = False
            with self._lock:
                owns_motion_slot = (
                    motion_generation == self._motion_generation
                    and self._motion_thread is threading.current_thread()
                )
                terminal_current = (
                    owns_motion_slot
                    and connection_generation == self._generation
                    and expected_controller is self._controller
                    and not self._closed
                    and self._connected
                )
                with self._motion_cancel_lock:
                    explicit_reason = self._motion_stop_reasons.pop(
                        motion_generation, None
                    )
                if owns_motion_slot:
                    self._motion_active = False
                    self._motion_thread = None
                if reason is None:
                    reason = explicit_reason
                if reason is None and not terminal_current:
                    reason = "disconnected"
                if reason is None:
                    reason = "manual"
                if error_payload is not None and terminal_current:
                    terminal_event = ServiceEvent(
                        "motion_error",
                        error_payload,
                        motion_generation,
                    )
                elif terminal_current:
                    terminal_event = ServiceEvent(
                        "motion_stopped",
                        reason,
                        motion_generation,
                    )
                if terminal_event is not None:
                    should_dispatch = self._reserve_event_locked(
                        terminal_event,
                        lambda: (
                            motion_generation == self._motion_generation
                            and connection_generation == self._generation
                            and expected_controller is self._controller
                            and not self._closed
                            and self._connected
                        ),
                    )
            if should_dispatch:
                self._drain_event_dispatch()

    def _setup_must_abort(self, generation: int) -> bool:
        with self._lock:
            return (
                generation != self._generation
                or self._closed
                or generation in self._setup_disconnected
            )

    def _connect_worker(
        self,
        generation: int,
        old_controller: Any | None = None,
    ) -> bool:
        if old_controller is not None and not self._disconnect_controller(
            old_controller
        ):
            self._retain_controller(old_controller)
            should_dispatch = False
            with self._lock:
                should_dispatch = self._mark_connection_unavailable_locked(
                    generation
                )
            if should_dispatch:
                self._drain_event_dispatch()
            return False
        if self._setup_must_abort(generation):
            return True
        try:
            controller = self._controller_factory(debug=False, auto_reconnect=True)
        except Exception:
            LOGGER.exception("Makcu controller factory failed")
            should_dispatch = False
            with self._lock:
                current = generation == self._generation and not self._closed
                if current:
                    self._controller = None
                    self._connected = False
                    self._disconnect_notified.add(generation)
                    should_dispatch = self._reserve_event_locked(
                        ServiceEvent(
                            "disconnected",
                            _CONNECTION_FAILURE_MESSAGE,
                        ),
                        lambda: (
                            generation == self._generation
                            and not self._closed
                            and self._controller is None
                            and not self._connected
                        ),
                    )
            if should_dispatch:
                self._drain_event_dispatch()
            return True

        try:
            controller.on_connection_change(
                lambda connected, g=generation: self._connection_changed(g, connected)
            )
            controller.enable_button_monitoring(True)
            controller.set_button_callback(
                lambda button, pressed, g=generation: self._button_event(
                    g, button, pressed
                )
            )
        except Exception:
            LOGGER.exception("Makcu controller setup failed")
            # Setup failures are equivalent to a failed connection, but the
            # exact newly-created controller must still be cleaned up.
            should_dispatch = False
            with self._lock:
                current = generation == self._generation and not self._closed
                if current:
                    already_notified = generation in self._disconnect_notified
                    self._setup_disconnected.add(generation)
                    self._disconnect_notified.add(generation)
                    self._controller = None
                    self._connected = False
                    if not already_notified:
                        should_dispatch = self._reserve_event_locked(
                            ServiceEvent(
                                "disconnected",
                                _CONNECTION_FAILURE_MESSAGE,
                            ),
                            lambda: (
                                generation == self._generation
                                and not self._closed
                                and self._controller is None
                                and not self._connected
                            ),
                        )
            if should_dispatch:
                self._drain_event_dispatch()
            cleanup_proven = self._disconnect_controller(controller)
            if not cleanup_proven:
                self._retain_controller(controller)
            return cleanup_proven

        if self._setup_must_abort(generation):
            cleanup_proven = self._disconnect_controller(controller)
            if not cleanup_proven:
                self._retain_controller(controller)
            return cleanup_proven

        diagnostics: list[str] = []
        for method_name in ("get_device_info", "get_firmware_version"):
            try:
                value = getattr(controller, method_name)()
            except Exception:
                continue
            if value is not None:
                diagnostics.append(str(value))

        # Installing the active controller is one lock-protected decision.
        # A setup-time false signal either set the abort flag above or can only
        # arrive after this assignment, in which case _connection_changed()
        # owns the active-generation transition.
        with self._lock:
            can_install = (
                generation == self._generation
                and not self._closed
                and generation not in self._setup_disconnected
            )
            if can_install:
                self._controller = controller
                self._connected = True
                installed_transition = self._connection_transition

        if not can_install:
            cleanup_proven = self._disconnect_controller(controller)
            if not cleanup_proven:
                self._retain_controller(controller)
            return cleanup_proven

        try:
            self._start_health_worker(generation, controller)
        except Exception:
            LOGGER.exception("Makcu health worker could not start")
            self._signal_motion_cancellation("disconnected")
            should_dispatch = False
            with self._lock:
                if (
                    generation == self._generation
                    and controller is self._controller
                ):
                    self._invalidate_motion_locked("disconnected")
                    should_dispatch = self._mark_connection_unavailable_locked(
                        generation
                    )
            with self._move_barrier:
                cleanup_proven = self._disconnect_controller(controller)
            if cleanup_proven:
                with self._lock:
                    self._retained_controllers = [
                        retained
                        for retained in self._retained_controllers
                        if retained is not controller
                    ]
            else:
                self._retain_controller(controller)
            if should_dispatch:
                self._drain_event_dispatch()
            return cleanup_proven

        should_dispatch = False
        with self._lock:
            can_publish = (
                generation == self._generation
                and not self._closed
                and controller is self._controller
                and self._connected
                and installed_transition == self._connection_transition
            )
            if can_publish:
                should_dispatch = self._reserve_event_locked(
                    ServiceEvent("connected", " | ".join(diagnostics) or None),
                    lambda: (
                        generation == self._generation
                        and not self._closed
                        and controller is self._controller
                        and self._connected
                        and installed_transition == self._connection_transition
                    ),
                )
        if should_dispatch:
            self._drain_event_dispatch()
        return True

    def _connection_changed(self, generation: int, connected: bool) -> None:
        connected = bool(connected)
        event: ServiceEvent | None = None
        stop_for_disconnect = False
        should_dispatch = False
        disconnect_transition: int | None = None
        with self._lock:
            if generation != self._generation or self._closed:
                return
            if self._controller is None:
                if not connected:
                    self._setup_disconnected.add(generation)
                    self._connected = False
                    if generation not in self._disconnect_notified:
                        self._disconnect_notified.add(generation)
                        event = ServiceEvent("disconnected")
                        should_dispatch = self._reserve_event_locked(
                            event,
                            lambda: (
                                generation == self._generation
                                and not self._closed
                                and not self._connected
                            ),
                        )
            elif connected:
                if not self._connected:
                    self._connected = True
                    self._connection_transition += 1
                    connection_transition = self._connection_transition
                    self._disconnect_notified.discard(generation)
                    event = ServiceEvent("reconnected")
                    should_dispatch = self._reserve_event_locked(
                        event,
                        lambda: (
                            generation == self._generation
                            and not self._closed
                            and self._connected
                            and connection_transition
                            == self._connection_transition
                        ),
                    )
            else:
                was_connected = self._connected
                self._connected = False
                stop_for_disconnect = was_connected and self._motion_active
                if was_connected:
                    self._connection_transition += 1
                    disconnect_transition = self._connection_transition
                    self._disconnect_notified.add(generation)
        if stop_for_disconnect:
            self.stop_motion(reason="disconnected")
        if disconnect_transition is not None:
            with self._lock:
                still_current_loss = (
                    generation == self._generation
                    and not self._closed
                    and not self._connected
                    and disconnect_transition == self._connection_transition
                    and generation in self._disconnect_notified
                )
                if still_current_loss:
                    event = ServiceEvent("disconnected")
                    should_dispatch = self._reserve_event_locked(
                        event,
                        lambda: (
                            generation == self._generation
                            and not self._closed
                            and not self._connected
                            and disconnect_transition
                            == self._connection_transition
                        ),
                    )
        if should_dispatch:
            self._drain_event_dispatch()

    @staticmethod
    def _normalize_button(button: Any) -> str | None:
        if isinstance(button, str):
            for name in BUTTON_NAMES.values():
                if button.strip().lower() == name.lower():
                    return name
            return None
        try:
            return BUTTON_NAMES[button]
        except (KeyError, TypeError):
            try:
                return BUTTON_NAMES[MouseButton(button)]
            except (ValueError, KeyError, TypeError):
                return None

    def _button_event(
        self,
        generation_or_button: Any,
        button_or_pressed: Any,
        pressed: Any = _MISSING,
    ) -> None:
        if pressed is _MISSING:
            with self._lock:
                generation = self._generation
            button = generation_or_button
            pressed = button_or_pressed
        else:
            generation = generation_or_button
            button = button_or_pressed
        name = self._normalize_button(button)
        if name is None:
            return
        should_dispatch = False
        with self._lock:
            if (
                generation != self._generation
                or self._closed
                or not self._connected
            ):
                return
            connection_transition = self._connection_transition
            should_dispatch = self._reserve_event_locked(
                ServiceEvent("button", (name, bool(pressed))),
                lambda: (
                    generation == self._generation
                    and not self._closed
                    and self._connected
                    and connection_transition == self._connection_transition
                ),
            )
        if should_dispatch:
            self._drain_event_dispatch()

    def reconnect(self) -> int | None:
        """Invalidate immediately; serialize cleanup and replacement off-thread."""
        reason = "disconnected"
        self._signal_motion_cancellation(reason)
        with self._lock:
            if self._closed:
                return None
            self._invalidate_motion_locked(reason)
            self._retain_controller_locked(self._controller)
            generation, should_dispatch = self._begin_connection_locked()
            self._connection_request_generation = generation
            thread = self._reserve_connection_worker_locked()
        self._launch_connection_worker(thread)
        if should_dispatch:
            self._drain_event_dispatch()
        return generation

    def close(self) -> None:
        """Invalidate all callbacks and asynchronously disconnect the device."""
        reason = "disconnected"
        self._signal_motion_cancellation(reason)
        with self._lock:
            if self._closed:
                return
            self._invalidate_motion_locked(reason)
            self._closed = True
            self._health_stop.set()
            self._generation += 1
            self._retain_controller_locked(self._controller)
            self._controller = None
            self._connected = False
            self._connection_request_generation = None
            thread = (
                self._reserve_connection_worker_locked()
                if self._retained_controllers
                else None
            )
        self._launch_connection_worker(thread)
