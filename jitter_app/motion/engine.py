"""Paired-pulse motion settings and pure motion engine."""

from dataclasses import dataclass
import math
from typing import Any, Mapping


MOTION_DEFAULTS = {
    "pulse_size_px": "2",
    "pulse_rate_hz": "60",
    "ramp_mode": "Smooth",
}
MOTION_LIMITS = {
    "pulse_size_px": (1.0, 8.0),
    "pulse_rate_hz": (20.0, 120.0),
}
RAMP_MODES = ("Instant", "Smooth")
PULSE_AXIS_DEGREES = 45.0
DEFAULT_SERVO_HZ = 1000
_DUE_EPSILON_S = 1e-12
MOTION_PRESETS = {
    "Soft": {"pulse_size_px": "1", "pulse_rate_hz": "30", "ramp_mode": "Smooth"},
    "Balanced": {"pulse_size_px": "2", "pulse_rate_hz": "60", "ramp_mode": "Smooth"},
    "Strong": {"pulse_size_px": "4", "pulse_rate_hz": "100", "ramp_mode": "Instant"},
}


@dataclass(frozen=True)
class MotionSettings:
    pulse_size_px: float = 2.0
    pulse_rate_hz: float = 60.0
    ramp_mode: str = "Smooth"


def _number(raw: Any, default: str, key: str) -> float:
    try:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        value = float(default)
    low, high = MOTION_LIMITS[key]
    return max(low, min(high, value))


def motion_settings_from_mapping(raw: Mapping[str, Any] | None) -> MotionSettings:
    values = dict(MOTION_DEFAULTS)
    if raw:
        values.update(raw)
    ramp_mode = values.get("ramp_mode")
    if ramp_mode not in RAMP_MODES:
        ramp_mode = MOTION_DEFAULTS["ramp_mode"]
    return MotionSettings(
        pulse_size_px=_number(
            values.get("pulse_size_px"), MOTION_DEFAULTS["pulse_size_px"], "pulse_size_px"
        ),
        pulse_rate_hz=_number(
            values.get("pulse_rate_hz"), MOTION_DEFAULTS["pulse_rate_hz"], "pulse_rate_hz"
        ),
        ramp_mode=ramp_mode,
    )


def _compact(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def motion_settings_to_mapping(settings: MotionSettings) -> dict[str, str]:
    return {
        "pulse_size_px": _compact(settings.pulse_size_px),
        "pulse_rate_hz": _compact(settings.pulse_rate_hz),
        "ramp_mode": settings.ramp_mode,
    }


@dataclass
class PairedPulseEngine:
    half_pulse_index: int = 0
    residual_x: float = 0.0
    residual_y: float = 0.0
    current_pair_x: int = 0
    current_pair_y: int = 0
    next_due_elapsed: float = 0.0

    def reset(self) -> None:
        self.half_pulse_index = 0
        self.residual_x = 0.0
        self.residual_y = 0.0
        self.current_pair_x = 0
        self.current_pair_y = 0
        self.next_due_elapsed = 0.0

    def step(self, settings: MotionSettings, dt: float, elapsed: float) -> tuple[int, int]:
        elapsed = max(0.0, float(elapsed))
        interval = 1.0 / (settings.pulse_rate_hz * 2.0)
        if elapsed + _DUE_EPSILON_S < self.next_due_elapsed:
            return 0, 0
        directions = (-1.0, 1.0, 1.0, -1.0)
        direction = directions[self.half_pulse_index % 4]
        if self.half_pulse_index % 2 == 0:
            ramp = 1.0
            if settings.ramp_mode == "Smooth":
                ramp = min(1.0, elapsed / 0.150)
            magnitude = settings.pulse_size_px * ramp
            angle = math.radians(PULSE_AXIS_DEGREES)
            total_x = magnitude * math.sin(angle) + self.residual_x
            total_y = magnitude * math.cos(angle) + self.residual_y
            self.current_pair_x = math.trunc(total_x)
            self.current_pair_y = math.trunc(total_y)
            self.residual_x = total_x - self.current_pair_x
            self.residual_y = total_y - self.current_pair_y
        report_x = max(-8, min(8, int(-direction * self.current_pair_x)))
        report_y = max(-8, min(8, int(direction * self.current_pair_y)))
        self.half_pulse_index += 1
        next_due_elapsed = self.next_due_elapsed + interval
        due_threshold = elapsed + _DUE_EPSILON_S
        if next_due_elapsed <= due_threshold:
            missed = math.floor((due_threshold - next_due_elapsed) / interval) + 1
            next_due_elapsed += max(1, missed) * interval
            if next_due_elapsed <= due_threshold:
                next_due_elapsed += interval
        self.next_due_elapsed = next_due_elapsed
        return report_x, report_y


@dataclass
class TriggerGate:
    trigger: str = "Left"
    modifier: str = "None"
    trigger_held: bool = False
    modifier_held: bool = False

    @property
    def active(self) -> bool:
        return self.trigger_held and (self.modifier == "None" or self.modifier_held)

    def update_button(self, name: str, pressed: bool) -> None:
        if name == self.trigger:
            self.trigger_held = pressed
        if self.modifier != "None" and name == self.modifier:
            self.modifier_held = pressed

    def configure(self, trigger: str, modifier: str) -> None:
        self.trigger = trigger
        self.modifier = modifier
        self.clear()

    def clear(self) -> None:
        self.trigger_held = False
        self.modifier_held = False
