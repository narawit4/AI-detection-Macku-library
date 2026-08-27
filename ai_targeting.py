"""Pure settings validation and target selection for AI aim mode."""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping


AIM_LIMITS: dict[str, tuple[float, float]] = {
    "confidence": (0.05, 0.95),
    "aim_strength": (0.05, 2.0),
    "smoothing": (0.0, 0.95),
    "max_step": (1.0, 127.0),
}

RESPONSE_CURVE_X = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_RESPONSE_CURVE = (0.0, 0.12, 0.35, 0.68, 1.0)


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


@dataclass(frozen=True)
class TargetSnapshot:
    sequence: int
    captured_at: float
    target_class: str
    aim_x: float
    aim_y: float


@dataclass(frozen=True)
class DetectionFrameSnapshot:
    sequence: int
    captured_at: float
    detections: tuple[Detection, ...]
    selected_index: int | None


@dataclass(frozen=True)
class DetectionAnalysis:
    target: TargetSnapshot | None
    frame: DetectionFrameSnapshot


TARGET_ASSOCIATION_RADIUS_PX = 48.0
TARGET_SWITCH_STABLE_DISPLACEMENT_PX = 18.0
TARGET_SWITCH_CONFIRMATION_COUNT = 3


@dataclass(frozen=True)
class TargetLockState:
    confirmed_target: TargetSnapshot | None = None
    pending_target: TargetSnapshot | None = None
    pending_count: int = 0


def _targets_are_near(
    first: TargetSnapshot,
    second: TargetSnapshot,
    radius: float,
) -> bool:
    return (
        first.target_class == second.target_class
        and math.hypot(
            first.aim_x - second.aim_x,
            first.aim_y - second.aim_y,
        ) <= radius
    )


def observe_target_lock(
    state: TargetLockState,
    candidate: TargetSnapshot | None,
) -> TargetLockState:
    confirmed = state.confirmed_target
    if candidate is None:
        return TargetLockState(confirmed_target=confirmed)
    if confirmed is None or _targets_are_near(
        confirmed,
        candidate,
        TARGET_ASSOCIATION_RADIUS_PX,
    ):
        return TargetLockState(confirmed_target=candidate)

    pending = state.pending_target
    pending_count = (
        state.pending_count + 1
        if pending is not None
        and _targets_are_near(
            pending,
            candidate,
            TARGET_SWITCH_STABLE_DISPLACEMENT_PX,
        )
        else 1
    )
    if pending_count >= TARGET_SWITCH_CONFIRMATION_COUNT:
        return TargetLockState(confirmed_target=candidate)
    return TargetLockState(confirmed, candidate, pending_count)


def target_lock_allows(
    state: TargetLockState,
    candidate: TargetSnapshot | None,
) -> bool:
    return (
        candidate is not None
        and state.pending_target is None
        and state.confirmed_target == candidate
    )


@dataclass(frozen=True)
class AimSettings:
    confidence: float = 0.35
    aim_strength: float = 0.35
    smoothing: float = 0.65
    max_step: int = 20
    response_curve: tuple[float, float, float, float, float] = DEFAULT_RESPONSE_CURVE


_DEFAULTS = AimSettings()


def _bounded_number(raw: Any, default: float, key: str) -> float:
    try:
        if isinstance(raw, bool):
            raise TypeError
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        value = default
    low, high = AIM_LIMITS[key]
    return max(low, min(high, value))


def _bounded_integer(raw: Any, default: int, key: str) -> int:
    try:
        if isinstance(raw, bool):
            raise TypeError
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        value = float(default)
    low, high = AIM_LIMITS[key]
    return int(max(low, min(high, value)))


def validated_response_curve(raw: Any) -> tuple[float, float, float, float, float]:
    if isinstance(raw, (str, bytes)):
        return DEFAULT_RESPONSE_CURVE
    try:
        values = tuple(raw)
    except TypeError:
        return DEFAULT_RESPONSE_CURVE
    if len(values) != len(RESPONSE_CURVE_X):
        return DEFAULT_RESPONSE_CURVE
    try:
        if any(isinstance(value, bool) for value in values):
            raise TypeError
        curve = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in curve):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_RESPONSE_CURVE
    if (
        curve[0] != 0.0
        or any(value < 0.0 or value > 1.0 for value in curve)
        or any(current < previous for previous, current in zip(curve, curve[1:]))
    ):
        return DEFAULT_RESPONSE_CURVE
    return curve[0], curve[1], curve[2], curve[3], curve[4]


def aim_settings_from_mapping(raw: Mapping[str, Any] | None) -> AimSettings:
    values = dict(raw or {})
    return AimSettings(
        confidence=_bounded_number(values.get("confidence"), _DEFAULTS.confidence, "confidence"),
        aim_strength=_bounded_number(values.get("aim_strength"), _DEFAULTS.aim_strength, "aim_strength"),
        smoothing=_bounded_number(values.get("smoothing"), _DEFAULTS.smoothing, "smoothing"),
        max_step=_bounded_integer(values.get("max_step"), _DEFAULTS.max_step, "max_step"),
        response_curve=validated_response_curve(values.get("response_curve")),
    )


def _compact(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def aim_settings_to_mapping(settings: AimSettings) -> dict[str, str | list[str]]:
    return {
        "confidence": _compact(settings.confidence),
        "aim_strength": _compact(settings.aim_strength),
        "smoothing": _compact(settings.smoothing),
        "max_step": str(settings.max_step),
        "response_curve": [_compact(value) for value in settings.response_curve],
    }


def response_curve_value(
    curve: tuple[float, ...], normalized_distance: float
) -> float:
    """Evaluate a validated response curve with monotone cubic interpolation."""
    try:
        x = float(normalized_distance)
    except (TypeError, ValueError, OverflowError):
        x = 0.0
    if not math.isfinite(x):
        x = 0.0
    x = max(RESPONSE_CURVE_X[0], min(RESPONSE_CURVE_X[-1], x))

    values = validated_response_curve(curve)
    secants = tuple(
        (right - left) / (x_right - x_left)
        for left, right, x_left, x_right in zip(
            values,
            values[1:],
            RESPONSE_CURVE_X,
            RESPONSE_CURVE_X[1:],
        )
    )
    tangents = [0.0] * len(values)
    tangents[0] = _endpoint_tangent(
        secants[0], secants[1], RESPONSE_CURVE_X[1] - RESPONSE_CURVE_X[0],
        RESPONSE_CURVE_X[2] - RESPONSE_CURVE_X[1],
    )
    tangents[-1] = _endpoint_tangent(
        secants[-1], secants[-2], RESPONSE_CURVE_X[-1] - RESPONSE_CURVE_X[-2],
        RESPONSE_CURVE_X[-2] - RESPONSE_CURVE_X[-3],
    )
    for index in range(1, len(values) - 1):
        left = secants[index - 1]
        right = secants[index]
        if left == 0.0 or right == 0.0 or left * right < 0.0:
            tangents[index] = 0.0
        else:
            left_width = RESPONSE_CURVE_X[index] - RESPONSE_CURVE_X[index - 1]
            right_width = RESPONSE_CURVE_X[index + 1] - RESPONSE_CURVE_X[index]
            first_weight = 2.0 * right_width + left_width
            second_weight = right_width + 2.0 * left_width
            tangents[index] = (first_weight + second_weight) / (
                first_weight / left + second_weight / right
            )

    segment = min(
        len(RESPONSE_CURVE_X) - 2,
        next(
            index
            for index in range(len(RESPONSE_CURVE_X) - 1)
            if x <= RESPONSE_CURVE_X[index + 1]
        ),
    )
    left_x = RESPONSE_CURVE_X[segment]
    width = RESPONSE_CURVE_X[segment + 1] - left_x
    t = (x - left_x) / width
    t_squared = t * t
    t_cubed = t_squared * t
    value = (
        (2.0 * t_cubed - 3.0 * t_squared + 1.0) * values[segment]
        + (t_cubed - 2.0 * t_squared + t) * width * tangents[segment]
        + (-2.0 * t_cubed + 3.0 * t_squared) * values[segment + 1]
        + (t_cubed - t_squared) * width * tangents[segment + 1]
    )
    return max(values[segment], min(values[segment + 1], value))


def _endpoint_tangent(
    edge_secant: float,
    neighbor_secant: float,
    edge_width: float,
    neighbor_width: float,
) -> float:
    if edge_secant == 0.0:
        return 0.0
    tangent = (
        (2.0 * edge_width + neighbor_width) * edge_secant
        - edge_width * neighbor_secant
    ) / (edge_width + neighbor_width)
    if tangent * edge_secant <= 0.0:
        return 0.0
    if edge_secant * neighbor_secant < 0.0 and abs(tangent) > abs(3.0 * edge_secant):
        return 3.0 * edge_secant
    return tangent


def detection_aim_point(detection: Detection) -> tuple[str, float, float] | None:
    if detection.class_id == 7:
        return "head", (detection.x1 + detection.x2) / 2.0, (
            detection.y1 + detection.y2
        ) / 2.0
    if detection.class_id == 0:
        return "player", (detection.x1 + detection.x2) / 2.0, (
            detection.y1 + (detection.y2 - detection.y1) * 0.20
        )
    return None


def analyze_detections(
    detections: Iterable[Detection],
    settings: AimSettings,
    *,
    sequence: int,
    captured_at: float,
    previous: TargetSnapshot | None = None,
) -> DetectionAnalysis:
    accepted = tuple(
        detection
        for detection in detections
        if detection.confidence >= settings.confidence
        and detection_aim_point(detection) is not None
    )
    candidates = [
        (index, point)
        for index, detection in enumerate(accepted)
        if (point := detection_aim_point(detection)) is not None
    ]
    heads = [item for item in candidates if item[1][0] == "head"]
    candidates = heads or [item for item in candidates if item[1][0] == "player"]
    selected_index = None
    target = None
    if candidates:
        target_class = candidates[0][1][0]
        origin = (160.0, 160.0)
        if previous is not None and previous.target_class == target_class:
            associated = [
                item for item in candidates
                if math.hypot(
                    item[1][1] - previous.aim_x,
                    item[1][2] - previous.aim_y,
                ) <= TARGET_ASSOCIATION_RADIUS_PX
            ]
            if associated:
                candidates = associated
                origin = (previous.aim_x, previous.aim_y)
        selected_index, selected = min(
            candidates,
            key=lambda item: math.hypot(
                item[1][1] - origin[0], item[1][2] - origin[1]
            ),
        )
        target = TargetSnapshot(sequence, captured_at, *selected)
    return DetectionAnalysis(
        target=target,
        frame=DetectionFrameSnapshot(
            sequence, captured_at, accepted, selected_index
        ),
    )


def select_target(
    detections: Iterable[Detection],
    settings: AimSettings,
    *,
    sequence: int,
    captured_at: float,
    previous: TargetSnapshot | None = None,
) -> TargetSnapshot | None:
    return analyze_detections(
        detections,
        settings,
        sequence=sequence,
        captured_at=captured_at,
        previous=previous,
    ).target


class AimMovementEngine:
    """Deterministic, stateful conversion of target snapshots into mouse deltas."""

    CENTER = 160.0
    DEAD_ZONE = 1.5
    MAX_AGE_S = 0.150
    MAX_ACCELERATION = 6.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_sequence = None
        self._previous_x = self._previous_y = 0.0
        self._fraction_x = self._fraction_y = 0.0

    def step(
        self,
        snapshot: TargetSnapshot | None,
        settings: AimSettings,
        now: float,
    ) -> tuple[int, int]:
        if snapshot is None:
            self.reset()
            return 0, 0
        if snapshot.sequence == self._last_sequence:
            return 0, 0
        self._last_sequence = snapshot.sequence
        age = max(0.0, now - snapshot.captured_at)
        if age > self.MAX_AGE_S:
            self.reset()
            return 0, 0

        error_x = snapshot.aim_x - self.CENTER
        error_y = snapshot.aim_y - self.CENTER
        if math.hypot(error_x, error_y) <= self.DEAD_ZONE:
            self._previous_x = self._previous_y = 0.0
            self._fraction_x = self._fraction_y = 0.0
            return 0, 0

        factor = 1.0 - settings.smoothing
        desired_x = error_x * settings.aim_strength
        desired_y = error_y * settings.aim_strength
        smoothed_x = self._previous_x + (desired_x - self._previous_x) * factor
        smoothed_y = self._previous_y + (desired_y - self._previous_y) * factor
        smoothed_x = max(self._previous_x - self.MAX_ACCELERATION,
                         min(self._previous_x + self.MAX_ACCELERATION, smoothed_x))
        smoothed_y = max(self._previous_y - self.MAX_ACCELERATION,
                         min(self._previous_y + self.MAX_ACCELERATION, smoothed_y))
        step = float(settings.max_step)
        clamped_x = max(-step, min(step, smoothed_x))
        clamped_y = max(-step, min(step, smoothed_y))
        self._previous_x, self._previous_y = clamped_x, clamped_y

        total_x = clamped_x + self._fraction_x
        total_y = clamped_y + self._fraction_y
        report_x = math.trunc(total_x)
        report_y = math.trunc(total_y)
        self._fraction_x = total_x - report_x
        self._fraction_y = total_y - report_y
        return report_x, report_y
