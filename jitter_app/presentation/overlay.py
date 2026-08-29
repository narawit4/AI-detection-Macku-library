"""Click-through, capture-excluded detection overlay for the primary display."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import logging
import threading
import tkinter as tk
from typing import Any, Callable

from jitter_app.ai.targeting import DetectionFrameSnapshot


LOGGER = logging.getLogger(__name__)

CAPTURE_SIZE = 320
OVERLAY_COLOR = "#ff2b2b"
MAX_FRAME_AGE_S = 0.150
_TRANSPARENT_KEY_CANDIDATES = ("#010203", "#010204", "#010205")

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
    label: str | None = None


@dataclass(frozen=True)
class OverlayStyle:
    """Immutable runtime-only visual choices for one overlay frame."""

    box_color: str = OVERLAY_COLOR
    show_heads: bool = True
    show_players: bool = True
    box_width: int = 2
    label_mode: str = "off"
    hud_visible: bool = True
    hud_corner: str = "top_left"
    hud_offset_x: int = 8
    hud_offset_y: int = 8
    hud_color: str = OVERLAY_COLOR
    hud_font_size: int = 10
    hud_show_fps: bool = True
    hud_show_provider: bool = True
    hud_show_zoom: bool = True
    hud_show_lock: bool = True


def _detection_label(class_id: int, confidence: float, mode: str) -> str | None:
    if mode not in {"class", "class_confidence"}:
        return None
    if class_id == 7:
        label = "HEAD"
    elif class_id == 0:
        label = "PLAYER"
    else:
        return None
    if mode == "class":
        return label
    return f"{label} {confidence:.0%}"


def _hud_position(
    style: OverlayStyle,
    screen_width: int,
    screen_height: int,
) -> tuple[int, int, str]:
    corner = style.hud_corner
    if corner not in {"top_left", "top_right", "bottom_left", "bottom_right"}:
        corner = "top_left"
    offset_x = max(0, min(screen_width, int(style.hud_offset_x)))
    offset_y = max(0, min(screen_height, int(style.hud_offset_y)))
    right = corner.endswith("right")
    bottom = corner.startswith("bottom")
    return (
        screen_width - offset_x if right else offset_x,
        screen_height - offset_y if bottom else offset_y,
        ("s" if bottom else "n") + ("e" if right else "w"),
    )


def project_overlay_boxes(
    snapshot: DetectionFrameSnapshot | None,
    now: float,
    *,
    show_heads: bool = True,
    show_players: bool = True,
    box_width: int = 2,
    label_mode: str = "off",
) -> tuple[OverlayBox, ...]:
    """Project a fresh immutable detector frame into canvas rectangles."""
    if snapshot is None or max(0.0, now - snapshot.captured_at) > MAX_FRAME_AGE_S:
        return ()
    box_width = max(1, min(8, int(box_width)))
    return tuple(
        OverlayBox(
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            box_width + 2 if index == snapshot.selected_index
            else box_width,
            _detection_label(
                detection.class_id,
                detection.confidence,
                label_mode,
            ),
        )
        for index, detection in enumerate(snapshot.detections)
        if (show_heads or detection.class_id != 7)
        and (show_players or detection.class_id != 0)
    )


def _selected_lock_label(
    snapshot: DetectionFrameSnapshot | None,
    now: float,
) -> str:
    if snapshot is None or max(0.0, now - snapshot.captured_at) > MAX_FRAME_AGE_S:
        return "NONE"
    selected_index = snapshot.selected_index
    if (
        selected_index is None
        or selected_index < 0
        or selected_index >= len(snapshot.detections)
    ):
        return "NONE"
    class_id = snapshot.detections[selected_index].class_id
    if class_id == 7:
        return "HEAD"
    if class_id == 0:
        return "PLAYER"
    return "NONE"


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
        self._capture_left = 0
        self._capture_top = 0
        self._screen_width = 0
        self._screen_height = 0
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
                screen_width = window.winfo_screenwidth()
                screen_height = window.winfo_screenheight()
                self._screen_width = screen_width
                self._screen_height = screen_height
                self._capture_left = (screen_width - CAPTURE_SIZE) // 2
                self._capture_top = (screen_height - CAPTURE_SIZE) // 2
                window.geometry(f"{screen_width}x{screen_height}+0+0")
                canvas = self._canvas_factory(
                    window,
                    width=screen_width,
                    height=screen_height,
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
        else:
            window = self._window
            screen_width = window.winfo_screenwidth()
            screen_height = window.winfo_screenheight()
            self._screen_width = screen_width
            self._screen_height = screen_height
            self._capture_left = (screen_width - CAPTURE_SIZE) // 2
            self._capture_top = (screen_height - CAPTURE_SIZE) // 2
            window.geometry(f"{screen_width}x{screen_height}+0+0")
            self._canvas.configure(
                width=screen_width,
                height=screen_height,
            )
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
        runtime: tuple[str, str, str] | None = None,
        style: OverlayStyle | None = None,
    ) -> None:
        self._require_main_thread()
        if style is None:
            style = OverlayStyle(
                box_color=color,
                show_heads=show_heads,
                hud_color=color,
            )
        color = style.box_color
        boxes = project_overlay_boxes(
            snapshot,
            now,
            show_heads=style.show_heads,
            show_players=style.show_players,
            box_width=style.box_width,
            label_mode=style.label_mode,
        )
        self.clear()
        visible_colors = [color]
        if runtime is not None and style.hud_visible:
            visible_colors.append(style.hud_color)
        self._keep_foregrounds_visible(*visible_colors)
        for box in boxes:
            self._canvas.create_rectangle(
                box.x1 + self._capture_left,
                box.y1 + self._capture_top,
                box.x2 + self._capture_left,
                box.y2 + self._capture_top,
                outline=color,
                width=box.width,
                tags=("detection",),
            )
            if box.label is not None:
                self._canvas.create_text(
                    box.x1 + self._capture_left,
                    box.y1 + self._capture_top - 2,
                    anchor="sw",
                    fill=color,
                    font=("Consolas", 9, "bold"),
                    text=box.label,
                    tags=("detection",),
                )
        if runtime is not None and style.hud_visible:
            fps, provider, zoom = runtime
            hud_lines = ["AI RUNTIME"]
            if style.hud_show_fps:
                hud_lines.append(f"FPS: {fps}")
            if style.hud_show_provider:
                hud_lines.append(f"PROVIDER: {provider}")
            if style.hud_show_zoom:
                hud_lines.append(f"ZOOM: {zoom}")
            if style.hud_show_lock:
                hud_lines.append(
                    f"LOCK: {_selected_lock_label(snapshot, now)}"
                )
            hud_x, hud_y, hud_anchor = _hud_position(
                style,
                self._screen_width,
                self._screen_height,
            )
            hud_item = self._canvas.create_text(
                hud_x,
                hud_y,
                anchor=hud_anchor,
                fill=style.hud_color,
                font=(
                    "Consolas",
                    max(8, min(24, int(style.hud_font_size))),
                    "bold",
                ),
                text="\n".join(hud_lines),
                tags=("runtime",),
            )
            self._keep_item_on_screen(hud_item)

    def _keep_item_on_screen(self, item_id: Any) -> None:
        bounds = self._canvas.bbox(item_id)
        if bounds is None:
            return
        left, top, right, bottom = bounds
        dx = -left if left < 0 else min(0, self._screen_width - right)
        dy = -top if top < 0 else min(0, self._screen_height - bottom)
        if dx or dy:
            self._canvas.move(item_id, dx, dy)

    def _keep_foregrounds_visible(self, *colors: str) -> None:
        folded = {color.casefold() for color in colors}
        if self._transparent_key.casefold() not in folded:
            return
        replacement = next(
            candidate
            for candidate in _TRANSPARENT_KEY_CANDIDATES
            if candidate.casefold() not in folded
        )
        self._canvas.configure(background=replacement)
        self._window.attributes("-transparentcolor", replacement)
        self._transparent_key = replacement

    def clear(self) -> None:
        self._require_main_thread()
        if self._canvas is not None:
            self._canvas.delete("detection")
            self._canvas.delete("runtime")

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
                canvas.delete("runtime")
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
