from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Iterable

import numpy as np

from ai_targeting import (
    AimSettings,
    Detection,
    DetectionAnalysis,
    DetectionFrameSnapshot,
    TargetSnapshot,
    analyze_detections,
    detection_aim_point,
)


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


@lru_cache(maxsize=16)
def _resize_plan(
    source_height: int,
    source_width: int,
    output_size: int,
) -> tuple[np.ndarray, ...]:
    source_x = np.linspace(0.0, source_width - 1, output_size)
    source_y = np.linspace(0.0, source_height - 1, output_size)
    x0 = np.floor(source_x).astype(np.intp)
    y0 = np.floor(source_y).astype(np.intp)
    plan = (
        x0,
        np.minimum(x0 + 1, source_width - 1),
        y0,
        np.minimum(y0 + 1, source_height - 1),
        (source_x - x0)[None, :, None],
        (source_y - y0)[:, None, None],
    )
    for values in plan:
        values.flags.writeable = False
    return plan


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
    x0, x1, y0, y1, wx, wy = _resize_plan(
        source_height, source_width, output_size
    )
    source = image.astype(np.float64)
    horizontal = (
        source[:, x0, :] * (1.0 - wx)
        + source[:, x1, :] * wx
    )
    blended = (
        horizontal[y0, :, :] * (1.0 - wy)
        + horizontal[y1, :, :] * wy
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


def compose_zoom_refinement(
    base: DetectionAnalysis,
    refined_detections: Iterable[Detection],
    transform: ZoomTransform,
    settings: AimSettings,
) -> DetectionAnalysis | None:
    selected_index = base.frame.selected_index
    if (
        base.target is None
        or selected_index is None
        or not 0 <= selected_index < len(base.frame.detections)
    ):
        return None
    seed = base.frame.detections[selected_index]
    if seed.class_id == 7:
        allowed_classes = {7}
    elif seed.class_id == 0:
        allowed_classes = {0, 7}
    else:
        return None

    margin_x = max(12.0, (seed.x2 - seed.x1) * 0.20)
    margin_y = max(12.0, (seed.y2 - seed.y1) * 0.20)
    compatible = []
    for detection in refined_detections:
        mapped = map_detection(detection, transform)
        if mapped is None or mapped.class_id not in allowed_classes:
            continue
        point = detection_aim_point(mapped)
        if point is None:
            continue
        _target_class, aim_x, aim_y = point
        if (
            seed.x1 - margin_x <= aim_x <= seed.x2 + margin_x
            and seed.y1 - margin_y <= aim_y <= seed.y2 + margin_y
        ):
            compatible.append(mapped)

    refined = analyze_detections(
        compatible,
        settings,
        sequence=base.frame.sequence,
        captured_at=base.frame.captured_at,
        previous=base.target,
    )
    if refined.target is None or refined.frame.selected_index is None:
        return None
    selected_refined = refined.frame.detections[
        refined.frame.selected_index
    ]
    composed = list(base.frame.detections)
    composed[selected_index] = selected_refined
    return DetectionAnalysis(
        refined.target,
        DetectionFrameSnapshot(
            base.frame.sequence,
            base.frame.captured_at,
            tuple(composed),
            selected_index,
        ),
    )
