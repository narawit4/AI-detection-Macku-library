"""Full-output DXCam capture for the AI aim worker."""

from collections.abc import Callable

import numpy as np


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


class DxcamCapture:
    """Own one DXCam camera and expose owned native RGB frames."""

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
            region=full_output_region(self._camera.width, self._camera.height),
            target_fps=self._target_fps,
        )
        self._capturing = True

    def read(self) -> np.ndarray | None:
        if not self._capturing or self._camera is None:
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
        return np.ascontiguousarray(frame.copy())

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
