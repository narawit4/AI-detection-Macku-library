"""Non-blocking hotkey cue playback."""

from __future__ import annotations

import logging
from pathlib import Path
import queue
import threading
from typing import Any, Callable


class _PygameBackend:
    def __init__(self, on_path: Path, off_path: Path) -> None:
        self._pygame = None
        self._channel = None
        self._sounds: dict[bool, Any] = {}
        self._mixer_cleanup_needed = False
        try:
            import pygame

            self._pygame = pygame
            self._mixer_cleanup_needed = True
            pygame.mixer.init()
            self._channel = pygame.mixer.Channel(0)
            self._sounds = {
                True: pygame.mixer.Sound(str(on_path)),
                False: pygame.mixer.Sound(str(off_path)),
            }
        except Exception:
            try:
                self.close()
            except Exception:
                logging.exception("Could not clean up partial hotkey sound backend")
            raise

    def play(self, enabled: bool) -> None:
        self._channel.play(self._sounds[bool(enabled)])

    def set_volume(self, volume: float) -> None:
        for sound in self._sounds.values():
            sound.set_volume(max(0.0, min(1.0, float(volume))))

    def close(self) -> None:
        channel, self._channel = self._channel, None
        pygame, self._pygame = self._pygame, None
        cleanup_needed, self._mixer_cleanup_needed = self._mixer_cleanup_needed, False
        self._sounds = {}
        error = None
        if channel is not None:
            try:
                channel.stop()
            except Exception as exc:
                error = exc
        if pygame is not None and cleanup_needed:
            try:
                pygame.mixer.quit()
            except Exception as exc:
                if error is None:
                    error = exc
                else:
                    logging.exception("Could not quit hotkey sound mixer")
        if error is not None:
            raise error


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
        self._close_lock = threading.Lock()
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
        with self._close_lock:
            if not self._closed.is_set():
                self._closed.set()
                self._queue.put(self._CLOSE)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    @staticmethod
    def _close_backend(backend: Any) -> None:
        try:
            backend.close()
        except Exception:
            logging.exception("Could not close hotkey sound player")

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
                    if backend is not None:
                        self._close_backend(backend)
                    backend = None
        finally:
            if backend is not None:
                self._close_backend(backend)
