"""Full-output DXCam capture for the AI aim worker."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


CENTER_320 = "center_320"
FULL_DISPLAY = "full_display"
CAPTURE_MODES = (CENTER_320, FULL_DISPLAY)


def validated_capture_mode(raw: Any) -> str:
    if type(raw) is not str or raw not in CAPTURE_MODES:
        raise ValueError("Unsupported AI capture mode")
    return raw


@dataclass(frozen=True)
class CapturedFrame:
    pixels: np.ndarray
    output_width: int
    output_height: int
    capture_left: int
    capture_top: int
    capture_width: int
    capture_height: int
    mode: str


def full_output_region(width: int, height: int) -> tuple[int, int, int, int]:
    """Return the complete native primary-output region."""
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
    ):
        raise ValueError("Primary output dimensions must be positive integers")
    return 0, 0, width, height


def centered_region(
    width: int,
    height: int,
    size: int = 320,
) -> tuple[int, int, int, int]:
    """Return a centered square region within the native primary output."""
    if (
        type(width) is not int
        or type(height) is not int
        or type(size) is not int
        or width <= 0
        or height <= 0
        or size <= 0
    ):
        raise ValueError("Capture dimensions must be positive integers")
    if width < size or height < size:
        raise ValueError("Primary output dimensions are smaller than capture size")
    left = (width - size) // 2
    top = (height - size) // 2
    return left, top, left + size, top + size


class DxcamCapture:
    """Own one DXCam camera and expose owned native RGB frames."""

    def __init__(
        self,
        camera_factory: Callable | None = None,
        mode: str = CENTER_320,
        target_fps: int = 120,
    ):
        self._camera_factory = camera_factory
        self._mode = validated_capture_mode(mode)
        self._target_fps = target_fps
        self._camera = None
        self._capturing = False
        self._active_geometry: tuple[int, int, int, int, int, int] | None = None

    def start(self) -> None:
        if self._capturing:
            return
        if self._camera is not None:
            self.close()
        if self._camera_factory is None:
            import dxcam

            self._camera_factory = dxcam.create
        self._camera = self._camera_factory(
            output_idx=0,
            output_color="RGB",
            processor_backend="numpy",
            max_buffer_len=2,
        )
        output_width = self._camera.width
        output_height = self._camera.height
        if (
            type(output_width) is not int
            or type(output_height) is not int
            or output_width <= 0
            or output_height <= 0
        ):
            raise ValueError("Primary output dimensions must be positive integers")
        region = (
            centered_region(output_width, output_height)
            if self._mode == CENTER_320
            else full_output_region(output_width, output_height)
        )
        self._camera.start(
            region=region,
            target_fps=self._target_fps,
        )
        left, top, right, bottom = region
        self._active_geometry = (
            output_width,
            output_height,
            left,
            top,
            right - left,
            bottom - top,
        )
        self._capturing = True

    def read(self) -> CapturedFrame | None:
        if (
            not self._capturing
            or self._camera is None
            or self._active_geometry is None
        ):
            return None
        frame = self._camera.get_latest_frame(copy=True)
        if frame is None:
            return None
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[0] <= 0
            or frame.shape[1] <= 0
            or frame.shape[2] != 3
            or frame.dtype != np.uint8
        ):
            raise ValueError("AI capture frame must be nonempty RGB uint8")
        (
            output_width,
            output_height,
            capture_left,
            capture_top,
            capture_width,
            capture_height,
        ) = self._active_geometry
        if frame.shape[:2] != (capture_height, capture_width):
            raise ValueError("AI capture frame must match capture region")
        pixels = np.ascontiguousarray(frame.copy())
        return CapturedFrame(
            pixels,
            output_width,
            output_height,
            capture_left,
            capture_top,
            capture_width,
            capture_height,
            self._mode,
        )

    def close(self) -> None:
        camera = self._camera
        if camera is None:
            self._active_geometry = None
            return
        self._camera = None
        was_capturing = self._capturing
        self._capturing = False
        self._active_geometry = None
        if was_capturing:
            try:
                camera.stop()
            except Exception:
                pass
        try:
            camera.release()
        except Exception:
            pass
