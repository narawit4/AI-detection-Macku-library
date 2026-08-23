"""Generation-safe Makcu connection lifecycle for the Jitter UI.

The service deliberately has no Tkinter dependency.  All callbacks become
small :class:`ServiceEvent` values and are delivered to the caller's sink;
the application layer is responsible for marshalling those values to Tk's
main thread.
"""

from dataclasses import dataclass
import threading
from typing import Any, Callable

from makcu import MouseButton, create_controller


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
    ) -> None:
        self._event_sink = event_sink
        self._controller_factory = controller_factory
        self._lock = threading.RLock()
        self._generation = 0
        self._controller: Any | None = None
        self._connected = False
        self._closed = False
        self._setup_disconnected: set[int] = set()
        self._disconnect_notified: set[int] = set()

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def controller(self) -> Any | None:
        with self._lock:
            return self._controller

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
                was_connected = self._connected
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
