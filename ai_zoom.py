from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ai_targeting import Detection, TargetSnapshot


FRAME_SIZE = 320
MAX_CENTER_DISTANCE = 96.0
_HEAD_TWO_X_MAX = 18.0
_HEAD_ONE_HALF_X_MAX = 32.0
_PLAYER_TWO_X_MAX = 64.0
_PLAYER_ONE_HALF_X_MAX = 112.0
_ZOOM_SOURCE_SIZES = {1.5: 213, 2.0: 160}


@dataclass(frozen=True)
class ZoomTransform:
    left: int
    top: int
    size: int
    factor: float


def select_zoom_factor(
    detection: Detection,
    target: TargetSnapshot | None,
) -> float:
    if target is None:
        return 1.0
    if (
        math.hypot(target.aim_x - 160.0, target.aim_y - 160.0)
        > MAX_CENTER_DISTANCE
    ):
        return 1.0
    height = detection.y2 - detection.y1
    if detection.class_id == 7:
        if height <= _HEAD_TWO_X_MAX:
            return 2.0
        if height <= _HEAD_ONE_HALF_X_MAX:
            return 1.5
    elif detection.class_id == 0:
        if height <= _PLAYER_TWO_X_MAX:
            return 2.0
        if height <= _PLAYER_ONE_HALF_X_MAX:
            return 1.5
    return 1.0


def resize_rgb_bilinear(
    image: np.ndarray,
    output_size: int = FRAME_SIZE,
) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] < 1
        or image.shape[1] < 1
        or image.dtype != np.uint8
    ):
        raise ValueError("Resize source must be a nonempty RGB uint8 array")
    if isinstance(output_size, bool) or int(output_size) != output_size:
        raise ValueError("Output size must be a positive integer")
    output_size = int(output_size)
    if output_size < 1:
        raise ValueError("Output size must be a positive integer")

    source_height, source_width = image.shape[:2]
    source_x = np.linspace(0.0, source_width - 1, output_size)
    source_y = np.linspace(0.0, source_height - 1, output_size)
    x0 = np.floor(source_x).astype(np.intp)
    y0 = np.floor(source_y).astype(np.intp)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx = (source_x - x0)[None, :, None]
    wy = (source_y - y0)[:, None, None]
    top_left = image[y0[:, None], x0[None, :]].astype(np.float32)
    top_right = image[y0[:, None], x1[None, :]].astype(np.float32)
    bottom_left = image[y1[:, None], x0[None, :]].astype(np.float32)
    bottom_right = image[y1[:, None], x1[None, :]].astype(np.float32)
    blended = (
        top_left * (1.0 - wx) * (1.0 - wy)
        + top_right * wx * (1.0 - wy)
        + bottom_left * (1.0 - wx) * wy
        + bottom_right * wx * wy
    )
    rounded = np.floor(np.clip(blended, 0.0, 255.0) + 0.5)
    return np.ascontiguousarray(rounded.astype(np.uint8))


def build_zoom_input(
    frame: np.ndarray,
    target: TargetSnapshot,
    factor: float,
) -> tuple[np.ndarray, ZoomTransform]:
    if not isinstance(frame, np.ndarray) or frame.shape != (320, 320, 3):
        raise ValueError("Zoom source must be RGB 320x320x3")
    if frame.dtype != np.uint8:
        raise ValueError("Zoom source must use uint8 pixels")
    try:
        size = _ZOOM_SOURCE_SIZES[float(factor)]
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Zoom factor must be 1.5 or 2.0") from error
    left = max(0, min(320 - size, round(target.aim_x - size / 2)))
    top = max(0, min(320 - size, round(target.aim_y - size / 2)))
    transform = ZoomTransform(left, top, size, float(factor))
    crop = frame[top:top + size, left:left + size]
    return resize_rgb_bilinear(crop), transform


def _map_coordinate(value: float, origin: int, scale: float) -> float:
    return max(0.0, min(320.0, origin + value * scale))


def map_detection(
    detection: Detection,
    transform: ZoomTransform,
) -> Detection | None:
    scale = transform.size / 320.0
    x1 = _map_coordinate(detection.x1, transform.left, scale)
    y1 = _map_coordinate(detection.y1, transform.top, scale)
    x2 = _map_coordinate(detection.x2, transform.left, scale)
    y2 = _map_coordinate(detection.y2, transform.top, scale)
    if x2 <= x1 or y2 <= y1:
        return None
    return Detection(
        x1, y1, x2, y2, detection.confidence, detection.class_id
    )


def map_target(
    target: TargetSnapshot,
    transform: ZoomTransform,
) -> TargetSnapshot:
    scale = transform.size / 320.0
    return TargetSnapshot(
        target.sequence,
        target.captured_at,
        target.target_class,
        _map_coordinate(target.aim_x, transform.left, scale),
        _map_coordinate(target.aim_y, transform.top, scale),
    )
