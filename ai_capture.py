"""Centered DXCam capture for the AI aim worker."""

from collections.abc import Callable

import numpy as np


def centered_region(width: int, height: int, size: int = 320) -> tuple[int, int, int, int]:
    """Return a square region centered on the primary output."""
    size = int(size)
    if size <= 0:
        raise ValueError("Capture region size must be positive")
    width = int(width)
    height = int(height)
    if width < size or height < size:
        raise ValueError("Primary output is smaller than the AI capture region")
    left = (width - size) // 2
    top = (height - size) // 2
    return left, top, left + size, top + size


class DxcamCapture:
    """Own one DXCam camera and expose owned RGB 320x320 frames."""

    def __init__(self, camera_factory: Callable | None = None, target_fps: int = 120):
        self._camera_factory = camera_factory
        self._target_fps = target_fps
        self._camera = None
        self._capturing = False

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
        self._camera.start(
            region=centered_region(self._camera.width, self._camera.height),
            target_fps=self._target_fps,
        )
        self._capturing = True

    def read(self) -> np.ndarray | None:
        if not self._capturing or self._camera is None:
            return None
        frame = self._camera.get_latest_frame(copy=True)
        if getattr(frame, "shape", None) != (320, 320, 3):
            return None
        return np.array(frame, dtype=np.uint8, copy=True)

    def close(self) -> None:
        camera = self._camera
        if camera is None:
            return
        self._camera = None
        was_capturing = self._capturing
        self._capturing = False
        if was_capturing:
            try:
                camera.stop()
            except Exception:
                pass
        try:
            camera.release()
        except Exception:
            pass
