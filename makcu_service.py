"""Generation-safe Makcu connection lifecycle for the Jitter UI.

The service deliberately has no Tkinter dependency.  All callbacks become
small :class:`ServiceEvent` values and are delivered to the caller's sink;
the application layer is responsible for marshalling those values to Tk's
main thread.
"""

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

from makcu import MouseButton, create_controller
from motion import SmoothMotionEngine


_MISSING = object()


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
        engine_factory: Callable[[], Any] = SmoothMotionEngine,
    ) -> None:
        self._event_sink = event_sink
        self._controller_factory = controller_factory
        self._engine_factory = engine_factory
        self._lock = threading.RLock()
        self._generation = 0
        self._controller: Any | None = None
        self._connected = False
        self._closed = False
        self._setup_disconnected: set[int] = set()
        self._disconnect_notified: set[int] = set()
        self._motion_generation = 0
        self._motion_stop = threading.Event()
        self._motion_thread: threading.Thread | None = None
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

    def _begin_connection(self) -> int:
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._controller = None
            self._connected = False
            self._setup_disconnected.discard(generation)
            self._disconnect_notified.discard(generation)
        self._emit(ServiceEvent("connecting"))
        return generation

    def connect(self) -> int | None:
        """Start a fresh connection worker and return its generation."""
        with self._lock:
            if self._closed:
                return None
        with self._lock:
            has_active_controller = self._controller is not None
        if has_active_controller:
            return self.reconnect()
        generation = self._begin_connection()
        self._start_connection_worker(generation)
        return generation

    def _start_connection_worker(self, generation: int) -> None:
        thread = threading.Thread(
            target=self._connect_worker,
            args=(generation,),
            name=f"MakcuConnect-{generation}",
            daemon=True,
        )
        thread.start()

    def _start_disconnect_worker(self, controller: Any) -> None:
        thread = threading.Thread(
            target=self._disconnect_controller,
            args=(controller,),
            name="MakcuDisconnect",
            daemon=True,
        )
        thread.start()

    @staticmethod
    def _disconnect_controller(controller: Any) -> None:
        try:
            controller.disconnect()
        except Exception:
            pass

    def start_motion(
        self,
        settings_provider: Callable[[], Any],
        duration_s: float | None = None,
    ) -> bool:
        """Start one interruptible movement worker for the active controller."""
        with self._lock:
            if self._closed or not self._connected or self._controller is None:
                return False
            if self._motion_active:
                return True
            try:
                duration = None if duration_s is None else max(0.0, float(duration_s))
            except (TypeError, ValueError):
                return False
            self._motion_generation += 1
            motion_generation = self._motion_generation
            connection_generation = self._generation
            stop_event = threading.Event()
            self._motion_stop = stop_event
            self._motion_stop_reasons[motion_generation] = None
            self._motion_active = True
            thread = threading.Thread(
                target=self._motion_worker,
                args=(
                    motion_generation,
                    connection_generation,
                    stop_event,
                    settings_provider,
                    duration,
                ),
                name=f"MakcuMotion-{motion_generation}",
                daemon=True,
            )
            self._motion_thread = thread
        try:
            thread.start()
        except Exception:
            with self._lock:
                if self._motion_generation == motion_generation:
                    self._motion_active = False
                    self._motion_thread = None
                    self._motion_stop_reasons.pop(motion_generation, None)
            return False
        return True

    def stop_motion(self, reason: str = "manual") -> None:
        """Request an immediate stop without waiting for the worker thread."""
        reason = str(reason or "manual")
        with self._lock:
            if not self._motion_active:
                return
            generation = self._motion_generation
            if self._motion_stop_reasons.get(generation) is None:
                self._motion_stop_reasons[generation] = reason
            self._motion_stop.set()

    def join_motion(self, timeout: float | None = None) -> None:
        """Bounded lifecycle/testing helper; do not call this from the Tk thread."""
        with self._lock:
            thread = self._motion_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _motion_worker(
        self,
        motion_generation: int,
        connection_generation: int,
        stop_event: threading.Event,
        settings_provider: Callable[[], Any],
        duration_s: float | None,
    ) -> None:
        engine: Any | None = None
        reason: str | None = None
        error_payload: str | None = None
        started = time.perf_counter()
        previous_tick = started
        try:
            try:
                engine = self._engine_factory()
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
                        reason = self._motion_stop_reasons.get(motion_generation) or "manual"
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
                    settings = settings_provider()
                    tick_started = time.perf_counter()
                    dt = max(0.0, min(tick_started - previous_tick, 0.1))
                    previous_tick = tick_started
                    report_x, report_y = engine.step(settings, dt, elapsed)
                except Exception as exc:
                    error_payload = f"{type(exc).__name__}: {exc}"
                    break
                if report_x or report_y:
                    # Serialize the stop check with the move call: once
                    # stop_motion() returns, this worker cannot send another
                    # report for this generation.
                    with self._lock:
                        if (
                            stop_event.is_set()
                            or motion_generation != self._motion_generation
                            or connection_generation != self._generation
                            or self._closed
                            or not self._connected
                            or controller is not self._controller
                        ):
                            reason = self._motion_stop_reasons.get(motion_generation) or "disconnected"
                            break
                        try:
                            controller.move(report_x, report_y)
                        except Exception as exc:
                            error_payload = f"{type(exc).__name__}: {exc}"
                            break
                try:
                    settings_rate = getattr(settings, "update_rate_hz", 20.0)
                    interval = 1.0 / max(float(settings_rate), 20.0)
                except Exception as exc:
                    error_payload = f"{type(exc).__name__}: {exc}"
                    break
                stop_event.wait(max(0.0, interval - (time.perf_counter() - tick_started)))
        finally:
            with self._lock:
                current = (
                    motion_generation == self._motion_generation
                    and self._motion_thread is threading.current_thread()
                )
                explicit_reason = self._motion_stop_reasons.pop(motion_generation, None)
                if current:
                    self._motion_active = False
                    self._motion_thread = None
                if reason is None:
                    reason = explicit_reason
                if reason is None and not current:
                    reason = "disconnected"
                if reason is None:
                    reason = "manual"
            if error_payload is not None:
                self._emit(ServiceEvent("motion_error", error_payload))
            elif current:
                self._emit(ServiceEvent("motion_stopped", reason))

    def _setup_must_abort(self, generation: int) -> bool:
        with self._lock:
            return (
                generation != self._generation
                or self._closed
                or generation in self._setup_disconnected
            )

    def _connect_worker(self, generation: int) -> None:
        try:
            controller = self._controller_factory(debug=False, auto_reconnect=True)
        except Exception as exc:
            with self._lock:
                current = generation == self._generation and not self._closed
                if current:
                    self._controller = None
                    self._connected = False
                    self._emit(
                        ServiceEvent(
                            "disconnected",
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
            return

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
        except Exception as exc:
            # Setup failures are equivalent to a failed connection, but the
            # exact newly-created controller must still be cleaned up.
            self._disconnect_controller(controller)
            with self._lock:
                current = generation == self._generation and not self._closed
                if current:
                    self._controller = None
                    self._connected = False
                    self._emit(
                        ServiceEvent(
                            "disconnected",
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
            return

        if self._setup_must_abort(generation):
            self._disconnect_controller(controller)
            return

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
                self._emit(ServiceEvent("connected", " | ".join(diagnostics) or None))

        if not can_install:
            self._disconnect_controller(controller)
        return

    def _connection_changed(self, generation: int, connected: bool) -> None:
        connected = bool(connected)
        event: ServiceEvent | None = None
        wait_for_motion = False
        controller_present = False
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
            elif connected:
                if not self._connected:
                    self._connected = True
                    event = ServiceEvent("reconnected")
            else:
                controller_present = True
                was_connected = self._connected
                wait_for_motion = was_connected and self._motion_active
        if wait_for_motion:
            self.stop_motion(reason="disconnected")
            self.join_motion(1.0)
        if not connected and controller_present:
            with self._lock:
                if generation != self._generation or self._closed:
                    return
                self._connected = False
                if was_connected and generation not in self._disconnect_notified:
                    self._disconnect_notified.add(generation)
                    event = ServiceEvent("disconnected")
        if event is not None:
            self._emit(event)

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
        with self._lock:
            if generation != self._generation or self._closed:
                return
            self._emit(ServiceEvent("button", (name, bool(pressed))))

    def reconnect(self) -> int | None:
        """Invalidate the current generation, clean it up, and reconnect."""
        self.stop_motion(reason="disconnected")
        with self._lock:
            if self._closed:
                return None
            old_controller = self._controller
        generation = self._begin_connection()
        if old_controller is not None:
            self._start_disconnect_worker(old_controller)
        self._start_connection_worker(generation)
        return generation

    def close(self) -> None:
        """Invalidate all callbacks and asynchronously disconnect the device."""
        self.stop_motion(reason="disconnected")
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            controller = self._controller
            self._controller = None
            self._connected = False
        if controller is not None:
            self._start_disconnect_worker(controller)
