from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

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
STABLE_DISPLACEMENT_PX = 18.0
STABLE_CONFIRMATION_COUNT = 2
RECOIL_COOLDOWN_SECONDS = 0.100


def _policy_scale(frame_width: int, frame_height: int) -> float:
    return FRAME_SIZE / max(frame_width, frame_height)


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
        or previous.frame_width != target.frame_width
        or previous.frame_height != target.frame_height
        or math.hypot(
            target.aim_x - previous.aim_x,
            target.aim_y - previous.aim_y,
        ) * _policy_scale(target.frame_width, target.frame_height)
        > STABLE_DISPLACEMENT_PX
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
    crop_width: int
    crop_height: int
    source_width: int
    source_height: int
    factor: float


def select_zoom_factor(
    detection: Detection,
    target: TargetSnapshot | None,
) -> float:
    if target is None:
        return 1.0
    policy_scale = _policy_scale(target.frame_width, target.frame_height)
    if (
        math.hypot(
            (target.aim_x - target.frame_width / 2.0) * policy_scale,
            (target.aim_y - target.frame_height / 2.0) * policy_scale,
        )
        > MAX_CENTER_DISTANCE
    ):
        return 1.0
    height = (detection.y2 - detection.y1) * policy_scale
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
    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[0] <= 0
        or frame.shape[1] <= 0
        or frame.shape[2] != 3
    ):
        raise ValueError("Zoom source must be a nonempty RGB frame")
    if frame.dtype != np.uint8:
        raise ValueError("Zoom source must use uint8 pixels")
    try:
        zoom_factor = float(factor)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Zoom factor must be 1.5 or 2.0") from error
    if zoom_factor not in (1.5, 2.0):
        raise ValueError("Zoom factor must be 1.5 or 2.0")
    source_height, source_width = frame.shape[:2]
    if (
        target.frame_width != source_width
        or target.frame_height != source_height
    ):
        raise ValueError("Target dimensions must match zoom source")
    crop_width = max(1, math.floor(source_width / zoom_factor + 0.5))
    crop_height = max(1, math.floor(source_height / zoom_factor + 0.5))
    left = max(
        0,
        min(
            source_width - crop_width,
            math.floor(target.aim_x - crop_width / 2 + 0.5),
        ),
    )
    top = max(
        0,
        min(
            source_height - crop_height,
            math.floor(target.aim_y - crop_height / 2 + 0.5),
        ),
    )
    transform = ZoomTransform(
        left,
        top,
        crop_width,
        crop_height,
        source_width,
        source_height,
        zoom_factor,
    )
    crop = np.ascontiguousarray(
        frame[top:top + crop_height, left:left + crop_width].copy()
    )
    return crop, transform


def _map_coordinate(value: float, origin: int, limit: int) -> float:
    return max(0.0, min(float(limit), origin + value))


def map_detection(
    detection: Detection,
    transform: ZoomTransform,
) -> Detection | None:
    x1 = _map_coordinate(
        detection.x1, transform.left, transform.source_width
    )
    y1 = _map_coordinate(
        detection.y1, transform.top, transform.source_height
    )
    x2 = _map_coordinate(
        detection.x2, transform.left, transform.source_width
    )
    y2 = _map_coordinate(
        detection.y2, transform.top, transform.source_height
    )
    if x2 <= x1 or y2 <= y1:
        return None
    return Detection(
        x1, y1, x2, y2, detection.confidence, detection.class_id
    )


def map_target(
    target: TargetSnapshot,
    transform: ZoomTransform,
) -> TargetSnapshot:
    return TargetSnapshot(
        target.sequence,
        target.captured_at,
        target.target_class,
        _map_coordinate(
            target.aim_x, transform.left, transform.source_width
        ),
        _map_coordinate(
            target.aim_y, transform.top, transform.source_height
        ),
        transform.source_width,
        transform.source_height,
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

    policy_scale = _policy_scale(
        base.frame.frame_width, base.frame.frame_height
    )
    fixed_source_margin = 12.0 / policy_scale
    margin_x = max(fixed_source_margin, (seed.x2 - seed.x1) * 0.20)
    margin_y = max(fixed_source_margin, (seed.y2 - seed.y1) * 0.20)
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
        base.frame.frame_width,
        base.frame.frame_height,
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
            base.frame.frame_width,
            base.frame.frame_height,
        ),
    )
