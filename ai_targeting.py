"""Pure settings validation and target selection for AI aim mode."""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping


AIM_LIMITS: dict[str, tuple[float, float]] = {
    "confidence": (0.0, 1.0),
    "aim_strength": (0.0, 2.0),
    "smoothing": (0.0, 1.0),
    "max_step": (1.0, 127.0),
}


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
class AimSettings:
    confidence: float = 0.35
    aim_strength: float = 0.35
    smoothing: float = 0.65
    max_step: int = 20


_DEFAULTS = AimSettings()


def _bounded_number(raw: Any, default: float, key: str) -> float:
    try:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        value = default
    low, high = AIM_LIMITS[key]
    return max(low, min(high, value))


def _bounded_integer(raw: Any, default: int, key: str) -> int:
    try:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        value = float(default)
    low, high = AIM_LIMITS[key]
    return int(max(low, min(high, value)))


def aim_settings_from_mapping(raw: Mapping[str, Any] | None) -> AimSettings:
    values = dict(raw or {})
    return AimSettings(
        confidence=_bounded_number(values.get("confidence"), _DEFAULTS.confidence, "confidence"),
        aim_strength=_bounded_number(values.get("aim_strength"), _DEFAULTS.aim_strength, "aim_strength"),
        smoothing=_bounded_number(values.get("smoothing"), _DEFAULTS.smoothing, "smoothing"),
        max_step=_bounded_integer(values.get("max_step"), _DEFAULTS.max_step, "max_step"),
    )


def _compact(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def aim_settings_to_mapping(settings: AimSettings) -> dict[str, str]:
    return {
        "confidence": _compact(settings.confidence),
        "aim_strength": _compact(settings.aim_strength),
        "smoothing": _compact(settings.smoothing),
        "max_step": str(settings.max_step),
    }


def _aim_point(detection: Detection) -> tuple[str, float, float] | None:
    if detection.class_id == 7:
        return "head", (detection.x1 + detection.x2) / 2.0, (
            detection.y1 + detection.y2
        ) / 2.0
    if detection.class_id == 0:
        return "player", (detection.x1 + detection.x2) / 2.0, (
            detection.y1 + (detection.y2 - detection.y1) * 0.20
        )
    return None


def select_target(
    detections: Iterable[Detection],
    settings: AimSettings,
    *,
    sequence: int,
    captured_at: float,
    previous: TargetSnapshot | None = None,
) -> TargetSnapshot | None:
    candidates = [
        point
        for detection in detections
        if detection.confidence >= settings.confidence
        if (point := _aim_point(detection)) is not None
    ]
    heads = [candidate for candidate in candidates if candidate[0] == "head"]
    candidates = heads or [candidate for candidate in candidates if candidate[0] == "player"]
    if not candidates:
        return None

    if previous is not None and previous.target_class == candidates[0][0]:
        associated = [
            candidate for candidate in candidates
            if math.hypot(candidate[1] - previous.aim_x, candidate[2] - previous.aim_y) <= 48.0
        ]
        if associated:
            candidates = associated
            origin = (previous.aim_x, previous.aim_y)
        else:
            origin = (160.0, 160.0)
    else:
        origin = (160.0, 160.0)

    selected = min(candidates, key=lambda candidate: math.hypot(
        candidate[1] - origin[0], candidate[2] - origin[1]
    ))
    return TargetSnapshot(sequence, captured_at, *selected)


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
