"""Edge-triggered polling for a Windows global virtual-key hotkey."""

from __future__ import annotations

import ctypes
import logging
import threading
from typing import Callable


class HotkeyEdgeDetector:
    def __init__(self):
        self._was_down = False

    def update(self, is_down: bool) -> bool:
        fired = bool(is_down) and not self._was_down
        self._was_down = bool(is_down)
        return fired

    def reset(self) -> None:
        self._was_down = False


def _default_key_state(vk: int) -> int:
    """Return the Windows async key state, resolving the API only when used."""
    return ctypes.windll.user32.GetAsyncKeyState(vk)


class HotkeyWatcher:
    """Poll a virtual key and invoke *callback* once for each press edge."""

    def __init__(
        self,
        vk: int,
        callback: Callable[[], None],
        key_state: Callable[[int], int] | None = None,
        poll_interval: float = 0.04,
    ):
        self._vk = int(vk)
        self._callback = callback
        self._key_state = key_state or _default_key_state
        self._poll_interval = max(0.0, float(poll_interval))
        self._detector = HotkeyEdgeDetector()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def set_vk(self, vk: int) -> None:
        with self._lock:
            self._vk = int(vk)
            self._detector.reset()

    def poll_once(self) -> None:
        with self._lock:
            is_down = bool(self._key_state(self._vk) & 0x8000)
            fired = self._detector.update(is_down)
        if not fired:
            return
        try:
            self._callback()
        except Exception:
            logging.exception("Global hotkey callback failed")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(self._poll_interval)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="JitterHotkey", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

