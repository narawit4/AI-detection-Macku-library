"""Pure settings validation and target selection for AI aim mode."""

from collections.abc import Sequence
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
TARGET_AREAS = ("head", "upper_body", "chest")


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


@dataclass(frozen=True)
class AimSettings:
    confidence: float = 0.35
    aim_strength: float = 0.35
    smoothing: float = 0.65
    max_step: int = 20
    response_curve: tuple[float, float, float, float, float] = DEFAULT_RESPONSE_CURVE
    target_area: str = "head"


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
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        return DEFAULT_RESPONSE_CURVE
    values = tuple(raw)
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


def validated_target_area(raw: Any) -> str:
    return raw if isinstance(raw, str) and raw in TARGET_AREAS else "head"


def aim_settings_from_mapping(raw: Mapping[str, Any] | None) -> AimSettings:
    values = dict(raw or {})
    return AimSettings(
        confidence=_bounded_number(values.get("confidence"), _DEFAULTS.confidence, "confidence"),
        aim_strength=_bounded_number(values.get("aim_strength"), _DEFAULTS.aim_strength, "aim_strength"),
        smoothing=_bounded_number(values.get("smoothing"), _DEFAULTS.smoothing, "smoothing"),
        max_step=_bounded_integer(values.get("max_step"), _DEFAULTS.max_step, "max_step"),
        response_curve=validated_response_curve(values.get("response_curve")),
        target_area=validated_target_area(values.get("target_area")),
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


def detection_aim_point(
    detection: Detection,
    target_area: str = "head",
) -> tuple[str, float, float] | None:
    area = validated_target_area(target_area)
    if detection.class_id == 7 and area == "head":
        return "head", (detection.x1 + detection.x2) / 2.0, (
            detection.y1 + detection.y2
        ) / 2.0
    if detection.class_id == 0:
        vertical_fraction = {
            "head": 0.20,
            "upper_body": 0.30,
            "chest": 0.42,
        }[area]
        return "player", (detection.x1 + detection.x2) / 2.0, (
            detection.y1 + (detection.y2 - detection.y1) * vertical_fraction
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
    """Select the nearest supported aim point from this frame only.

    ``previous`` remains accepted for API compatibility but intentionally has
    no influence on current-frame selection.
    """
    accepted = tuple(
        detection
        for detection in detections
        if detection.confidence >= settings.confidence
        and detection_aim_point(detection) is not None
    )
    candidates = [
        (index, point)
        for index, detection in enumerate(accepted)
        if (
            point := detection_aim_point(detection, settings.target_area)
        ) is not None
    ]
    selected_index = None
    target = None
    if candidates:
        selected_index, selected = min(
            candidates,
            key=lambda item: math.hypot(
                item[1][1] - 160.0, item[1][2] - 160.0
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
    MAX_ACCELERATION = 21_600.0
    MAX_SMOOTHING_TAU_S = 0.200
    MAX_DT_S = 0.100
    REFERENCE_RADIUS = math.hypot(CENTER, CENTER)

    def __init__(self, nominal_hz: float = 240.0) -> None:
        self._nominal_hz = nominal_hz
        self.reset()

    def reset(self) -> None:
        self._settled_sequence = None
        self._last_sequence = None
        self._remaining_x = self._remaining_y = 0.0
        self._velocity_x = self._velocity_y = 0.0
        self._fraction_x = self._fraction_y = 0.0
        self._previous_tick = None
        self._target_captured_at = None

    def step(
        self,
        snapshot: TargetSnapshot | None,
        settings: AimSettings,
        now: float,
    ) -> tuple[int, int]:
        if snapshot is None:
            self.reset()
            return 0, 0
        if snapshot.sequence == self._settled_sequence:
            return 0, 0
        fresh_sequence = snapshot.sequence != self._last_sequence
        target_captured_at = (
            snapshot.captured_at if fresh_sequence else self._target_captured_at
        )
        if now > target_captured_at + self.MAX_AGE_S:
            self.reset()
            return 0, 0

        if fresh_sequence:
            self._settled_sequence = None
            self._last_sequence = snapshot.sequence
            self._remaining_x = snapshot.aim_x - self.CENTER
            self._remaining_y = snapshot.aim_y - self.CENTER
            self._fraction_x = self._fraction_y = 0.0
            self._target_captured_at = snapshot.captured_at

        radius = math.hypot(self._remaining_x, self._remaining_y)
        if radius <= self.DEAD_ZONE:
            if fresh_sequence:
                self.reset()
            else:
                self._velocity_x = self._velocity_y = 0.0
                self._fraction_x = self._fraction_y = 0.0
            return 0, 0

        if self._previous_tick is None:
            dt = 1.0 / self._nominal_hz
        else:
            dt = max(0.0, min(self.MAX_DT_S, now - self._previous_tick))
        self._previous_tick = now

        normalized = min(1.0, radius / self.REFERENCE_RADIUS)
        curve_distance = (
            response_curve_value(settings.response_curve, normalized)
            * self.REFERENCE_RADIUS
        )
        reference_step = min(
            float(settings.max_step), curve_distance * settings.aim_strength
        )
        desired_speed = reference_step * 60.0
        desired_x = desired_speed * self._remaining_x / radius
        desired_y = desired_speed * self._remaining_y / radius

        if settings.smoothing > 0.0:
            tau = self.MAX_SMOOTHING_TAU_S * (settings.smoothing / 0.95) ** 2
            alpha = 1.0 - math.exp(-dt / tau)
        else:
            alpha = 1.0
        next_x = self._velocity_x + (desired_x - self._velocity_x) * alpha
        next_y = self._velocity_y + (desired_y - self._velocity_y) * alpha
        change_x = next_x - self._velocity_x
        change_y = next_y - self._velocity_y
        change = math.hypot(change_x, change_y)
        max_change = self.MAX_ACCELERATION * dt
        if change > max_change and change > 0.0:
            scale = max_change / change
            change_x *= scale
            change_y *= scale
        self._velocity_x += change_x
        self._velocity_y += change_y

        total_x = self._velocity_x * dt + self._fraction_x
        total_y = self._velocity_y * dt + self._fraction_y
        candidate_x = math.trunc(total_x)
        candidate_y = math.trunc(total_y)
        self._fraction_x = total_x - candidate_x
        self._fraction_y = total_y - candidate_y
        report_x = self._clamp_report(candidate_x, self._remaining_x, settings.max_step)
        report_y = self._clamp_report(candidate_y, self._remaining_y, settings.max_step)
        self._remaining_x -= report_x
        self._remaining_y -= report_y
        if math.hypot(self._remaining_x, self._remaining_y) <= self.DEAD_ZONE:
            settled_sequence = self._last_sequence
            self.reset()
            self._settled_sequence = settled_sequence
        return report_x, report_y

    @staticmethod
    def _clamp_report(candidate: int, remaining: float, max_step: int) -> int:
        candidate = max(-int(max_step), min(int(max_step), candidate))
        if remaining > 0.0:
            return max(0, min(candidate, math.floor(remaining)))
        if remaining < 0.0:
            return min(0, max(candidate, math.ceil(remaining)))
        return 0
