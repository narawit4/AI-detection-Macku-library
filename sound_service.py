"""Non-blocking hotkey cue playback."""

from __future__ import annotations

import logging
from pathlib import Path
import queue
import threading
from typing import Any, Callable


class _PygameBackend:
    def __init__(self, on_path: Path, off_path: Path) -> None:
        import pygame

        pygame.mixer.init()
        self._pygame = pygame
        self._channel = pygame.mixer.Channel(0)
        self._sounds = {
            True: pygame.mixer.Sound(str(on_path)),
            False: pygame.mixer.Sound(str(off_path)),
        }

    def play(self, enabled: bool) -> None:
        self._channel.play(self._sounds[bool(enabled)])

    def set_volume(self, volume: float) -> None:
        for sound in self._sounds.values():
            sound.set_volume(max(0.0, min(1.0, float(volume))))

    def close(self) -> None:
        self._channel.stop()
        self._pygame.mixer.quit()


class ToggleSoundPlayer:
    """Queue ON/OFF cues so audio initialization never blocks Tk."""

    _CLOSE = object()

    def __init__(
        self,
        sound_dir: Path,
        *,
        backend_factory: Callable[[Path, Path], Any] | None = None,
        enabled: bool = True,
        volume: int = 70,
    ) -> None:
        self._sound_dir = Path(sound_dir)
        self._backend_factory = backend_factory or _PygameBackend
        self._queue: queue.Queue[tuple[bool, int] | object] = queue.Queue()
        self._closed = threading.Event()
        self._settings_lock = threading.Lock()
        self._enabled = bool(enabled)
        self._volume = max(0, min(100, int(volume)))
        self._thread = threading.Thread(
            target=self._run,
            name="jitter-sound",
            daemon=True,
        )
        self._thread.start()

    def configure(self, *, enabled: bool, volume: int) -> None:
        with self._settings_lock:
            self._enabled = bool(enabled)
            self._volume = max(0, min(100, int(volume)))

    def play(self, enabled: bool, *, force: bool = False) -> None:
        with self._settings_lock:
            audible = self._enabled or bool(force)
            volume = self._volume
        if audible and not self._closed.is_set():
            self._queue.put((bool(enabled), volume))

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(self._CLOSE)
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        backend = None
        try:
            while True:
                item = self._queue.get()
                if item is self._CLOSE:
                    return
                cue, volume = item
                try:
                    if backend is None:
                        backend = self._backend_factory(
                            self._sound_dir / "ON.ogg",
                            self._sound_dir / "OFF.ogg",
                        )
                    backend.set_volume(volume / 100.0)
                    backend.play(bool(cue))
                except Exception:
                    logging.exception("Could not play hotkey sound")
                    backend = None
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    logging.exception("Could not close hotkey sound player")
