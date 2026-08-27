"""Click-through, capture-excluded detection overlay for the primary display."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import logging
import threading
import tkinter as tk
from typing import Any, Callable

from ai_targeting import DetectionFrameSnapshot


LOGGER = logging.getLogger(__name__)

OVERLAY_SIZE = 320
OVERLAY_COLOR = "#ff2b2b"
MAX_FRAME_AGE_S = 0.150

WDA_EXCLUDEFROMCAPTURE = 0x00000011
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000

_GWL_EXSTYLE = -20
_GA_ROOT = 2
_REQUIRED_EXTENDED_STYLES = (
    WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_NOACTIVATE
)


class OverlaySetupError(RuntimeError):
    """Raised when the overlay cannot meet its required safety contract."""


@dataclass(frozen=True)
class OverlayBox:
    x1: float
    y1: float
    x2: float
    y2: float
    width: int


def project_overlay_boxes(
    snapshot: DetectionFrameSnapshot | None,
    now: float,
    *,
    show_heads: bool = True,
) -> tuple[OverlayBox, ...]:
    """Project a fresh immutable detector frame into canvas rectangles."""
    if snapshot is None or max(0.0, now - snapshot.captured_at) > MAX_FRAME_AGE_S:
        return ()
    return tuple(
        OverlayBox(
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            4 if index == snapshot.selected_index else 2,
        )
        for index, detection in enumerate(snapshot.detections)
        if show_heads or detection.class_id != 7
    )


class Win32OverlayAdapter:
    """Applies all native properties required before an overlay is shown."""

    def __init__(
        self,
        *,
        user32: Any | None = None,
        get_last_error: Callable[[], int] = ctypes.get_last_error,
        set_last_error: Callable[[int], None] = ctypes.set_last_error,
    ) -> None:
        if user32 is None:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            long_ptr = ctypes.c_ssize_t
            user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.GetWindowLongPtrW.restype = long_ptr
            user32.SetWindowLongPtrW.argtypes = [
                wintypes.HWND,
                ctypes.c_int,
                long_ptr,
            ]
            user32.SetWindowLongPtrW.restype = long_ptr
            user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
            user32.GetAncestor.restype = wintypes.HWND
            user32.SetWindowDisplayAffinity.argtypes = [
                wintypes.HWND,
                wintypes.DWORD,
            ]
            user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
        self._user32 = user32
        self._get_last_error = get_last_error
        self._set_last_error = set_last_error

    def configure(self, hwnd: int) -> None:
        self._set_last_error(0)
        root_hwnd = self._user32.GetAncestor(hwnd, _GA_ROOT)
        if not root_hwnd:
            raise self._failure("GetAncestor", self._get_last_error())
        hwnd = int(root_hwnd)

        self._set_last_error(0)
        existing_styles = self._user32.GetWindowLongPtrW(hwnd, _GWL_EXSTYLE)
        error = self._get_last_error()
        if existing_styles == 0 and error:
            raise self._failure("GetWindowLongPtrW", error)

        self._set_last_error(0)
        previous_styles = self._user32.SetWindowLongPtrW(
            hwnd,
            _GWL_EXSTYLE,
            existing_styles | _REQUIRED_EXTENDED_STYLES,
        )
        error = self._get_last_error()
        if previous_styles == 0 and error:
            raise self._failure("SetWindowLongPtrW", error)

        self._set_last_error(0)
        if not self._user32.SetWindowDisplayAffinity(
            hwnd, WDA_EXCLUDEFROMCAPTURE
        ):
            raise self._failure(
                "SetWindowDisplayAffinity", self._get_last_error()
            )

    @staticmethod
    def _failure(operation: str, error: int) -> OverlaySetupError:
        return OverlaySetupError(
            f"{operation} failed (Windows error {error})"
        )


class DetectionOverlay:
    """Owns a lazily created Tk overlay and its deterministic lifecycle."""

    def __init__(
        self,
        root: tk.Misc,
        *,
        window_factory: Callable[[tk.Misc], Any] = tk.Toplevel,
        canvas_factory: Callable[..., Any] = tk.Canvas,
        win32_adapter: Any | None = None,
        transparent_key: str = "#010203",
    ) -> None:
        self._root = root
        self._window_factory = window_factory
        self._canvas_factory = canvas_factory
        self._win32 = win32_adapter or Win32OverlayAdapter()
        self._transparent_key = transparent_key
        self._window = None
        self._canvas = None
        self._visible = False
        self._closed = False

    @property
    def visible(self) -> bool:
        return self._visible

    def show(self) -> None:
        self._require_main_thread()
        if self._closed:
            raise OverlaySetupError("Overlay is closed")
        if self._window is None:
            window = self._window_factory(self._root)
            try:
                window.withdraw()
                window.overrideredirect(True)
                window.attributes("-topmost", True)
                window.attributes("-transparentcolor", self._transparent_key)
                left = (window.winfo_screenwidth() - OVERLAY_SIZE) // 2
                top = (window.winfo_screenheight() - OVERLAY_SIZE) // 2
                window.geometry(f"{OVERLAY_SIZE}x{OVERLAY_SIZE}+{left}+{top}")
                canvas = self._canvas_factory(
                    window,
                    width=OVERLAY_SIZE,
                    height=OVERLAY_SIZE,
                    background=self._transparent_key,
                    highlightthickness=0,
                )
                canvas.pack(fill="both", expand=True)
                window.update_idletasks()
                self._win32.configure(int(window.winfo_id()))
            except Exception:
                self._destroy_safely(window, "Overlay setup cleanup failed")
                raise
            self._window = window
            self._canvas = canvas
        window = self._window
        try:
            window.deiconify()
            window.lift()
        except Exception:
            self._window = None
            self._canvas = None
            self._visible = False
            try:
                window.withdraw()
            except Exception:
                LOGGER.exception("Overlay activation withdraw failed")
            self._destroy_safely(window, "Overlay activation cleanup failed")
            raise
        self._visible = True

    def render(
        self,
        snapshot: DetectionFrameSnapshot | None,
        *,
        now: float,
        color: str = OVERLAY_COLOR,
        show_heads: bool = True,
    ) -> None:
        self._require_main_thread()
        boxes = project_overlay_boxes(snapshot, now, show_heads=show_heads)
        self.clear()
        for box in boxes:
            self._canvas.create_rectangle(
                box.x1,
                box.y1,
                box.x2,
                box.y2,
                outline=color,
                width=box.width,
                tags=("detection",),
            )

    def clear(self) -> None:
        self._require_main_thread()
        if self._canvas is not None:
            self._canvas.delete("detection")

    def hide(self) -> None:
        self._require_main_thread()
        self.clear()
        if self._window is not None:
            self._window.withdraw()
        self._visible = False

    def close(self) -> None:
        self._require_main_thread()
        window, canvas = self._window, self._canvas
        self._window = None
        self._canvas = None
        self._visible = False
        self._closed = True
        if canvas is not None:
            try:
                canvas.delete("detection")
            except Exception:
                LOGGER.exception("Detection overlay canvas cleanup failed")
        if window is not None:
            self._destroy_safely(window, "Detection overlay window cleanup failed")

    @staticmethod
    def _require_main_thread() -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Detection overlay methods require the main thread")

    @staticmethod
    def _destroy_safely(window: Any, message: str) -> None:
        try:
            window.destroy()
        except Exception:
            LOGGER.exception(message)
