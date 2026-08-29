from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .resize import resize_rgb_bilinear
from .targeting import (
    AimSettings,
    Detection,
    DetectionAnalysis,
    DetectionFrameSnapshot,
    TargetSnapshot,
    detection_aim_point,
    validated_target_area,
)


FRAME_SIZE = 320
MAX_CENTER_DISTANCE = 96.0
_HEAD_TWO_X_MAX = 18.0
_HEAD_ONE_HALF_X_MAX = 32.0
_PLAYER_TWO_X_MAX = 64.0
_PLAYER_ONE_HALF_X_MAX = 112.0
_ZOOM_SOURCE_SIZES = {1.5: 213, 2.0: 160}
STABLE_DISPLACEMENT_PX = 18.0
STABLE_CONFIRMATION_COUNT = 2
RECOIL_COOLDOWN_SECONDS = 0.100


@dataclass(frozen=True)
class ZoomStabilityState:
    previous_base_target: TargetSnapshot | None = None
    stable_count: int = 0
    cooldown_until: float = 0.0


def _extended_zoom_cooldown(
    state: ZoomStabilityState,
    now: float,
) -> float:
    return max(
        state.cooldown_until,
        now + RECOIL_COOLDOWN_SECONDS,
    )


def observe_zoom_stability(
    state: ZoomStabilityState,
    target: TargetSnapshot | None,
    now: float,
) -> ZoomStabilityState:
    if target is None:
        return ZoomStabilityState(
            None,
            0,
            _extended_zoom_cooldown(state, now),
        )

    previous = state.previous_base_target
    unstable = (
        previous is None
        or previous.target_class != target.target_class
        or math.hypot(
            target.aim_x - previous.aim_x,
            target.aim_y - previous.aim_y,
        ) > STABLE_DISPLACEMENT_PX
    )
    if unstable:
        return ZoomStabilityState(
            target,
            1,
            _extended_zoom_cooldown(state, now),
        )
    return ZoomStabilityState(
        target,
        min(STABLE_CONFIRMATION_COUNT, state.stable_count + 1),
        state.cooldown_until,
    )


def record_zoom_refinement_miss(
    state: ZoomStabilityState,
    now: float,
) -> ZoomStabilityState:
    return ZoomStabilityState(
        state.previous_base_target,
        0,
        _extended_zoom_cooldown(state, now),
    )


def movement_is_confirmed(state: ZoomStabilityState) -> bool:
    return (
        state.previous_base_target is not None
        and state.stable_count >= STABLE_CONFIRMATION_COUNT
    )


def limit_zoom_factor(
    requested_factor: float,
    state: ZoomStabilityState,
    now: float,
) -> float:
    if requested_factor == 2.0 and (
        not movement_is_confirmed(state)
        or now < state.cooldown_until
    ):
        return 1.5
    return float(requested_factor)


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
        allowed_classes = (
            {0, 7}
            if validated_target_area(settings.target_area) == "head"
            else {0}
        )
    else:
        return None

    margin_x = max(12.0, (seed.x2 - seed.x1) * 0.20)
    margin_y = max(12.0, (seed.y2 - seed.y1) * 0.20)
    compatible = []
    for detection in refined_detections:
        mapped = map_detection(detection, transform)
        if (
            mapped is None
            or mapped.class_id not in allowed_classes
            or mapped.confidence < settings.confidence
        ):
            continue
        point = detection_aim_point(mapped, settings.target_area)
        if point is None:
            continue
        _target_class, aim_x, aim_y = point
        if (
            seed.x1 - margin_x <= aim_x <= seed.x2 + margin_x
            and seed.y1 - margin_y <= aim_y <= seed.y2 + margin_y
        ):
            compatible.append((mapped, point))

    if not compatible:
        return None
    selected_refined, selected_point = min(
        compatible,
        key=lambda item: math.hypot(
            item[1][1] - base.target.aim_x,
            item[1][2] - base.target.aim_y,
        ),
    )
    refined_target = TargetSnapshot(
        base.frame.sequence,
        base.frame.captured_at,
        *selected_point,
    )
    composed = list(base.frame.detections)
    composed[selected_index] = selected_refined
    return DetectionAnalysis(
        refined_target,
        DetectionFrameSnapshot(
            base.frame.sequence,
            base.frame.captured_at,
            tuple(composed),
            selected_index,
        ),
    )
