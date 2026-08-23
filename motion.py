"""Motion settings domain model and serialization helpers."""

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping


MOTION_DEFAULTS = {
    "motion_angle_deg": "90",
    "motion_strength_pps": "80",
    "jitter_enabled": True,
    "horizontal_jitter_pps": "55",
    "vertical_jitter_pps": "40",
    "smoothness_percent": "25",
    "update_rate_hz": "240",
    "ramp_up_ms": "80",
    "jitter_rate_hz": "14",
    "jitter_randomness_percent": "25",
    "jitter_axis_phase_deg": "90",
    "jitter_waveform": "Random blend",
    "max_step_px": "8",
    "acceleration_pps2": "2500",
    "deceleration_pps2": "3500",
    "motion_curve": "S-curve",
}

MOTION_LIMITS = {
    "motion_angle_deg": (0.0, 360.0),
    "motion_strength_pps": (0.0, 500.0),
    "horizontal_jitter_pps": (0.0, 500.0),
    "vertical_jitter_pps": (0.0, 500.0),
    "smoothness_percent": (1.0, 100.0),
    "update_rate_hz": (20.0, 500.0),
    "ramp_up_ms": (0.0, 2000.0),
    "jitter_rate_hz": (0.1, 60.0),
    "jitter_randomness_percent": (0.0, 100.0),
    "jitter_axis_phase_deg": (0.0, 360.0),
    "max_step_px": (1.0, 50.0),
    "acceleration_pps2": (1.0, 10000.0),
    "deceleration_pps2": (1.0, 10000.0),
}

MOTION_CURVES = ("Linear", "Ease-in", "S-curve")
JITTER_WAVEFORMS = ("Sine", "Triangle", "Square", "Random blend")

MOTION_PRESETS = {
    "Ultra Stable": {"motion_strength_pps": "15", "horizontal_jitter_pps": "0", "vertical_jitter_pps": "0", "smoothness_percent": "95", "ramp_up_ms": "400", "max_step_px": "1", "acceleration_pps2": "60", "deceleration_pps2": "160", "motion_curve": "S-curve", "jitter_enabled": False},
    "Soft": {"motion_strength_pps": "25", "horizontal_jitter_pps": "1", "vertical_jitter_pps": "0", "smoothness_percent": "90", "ramp_up_ms": "300", "max_step_px": "1", "acceleration_pps2": "80", "deceleration_pps2": "180", "motion_curve": "S-curve", "jitter_enabled": True},
    "Balanced": {"motion_strength_pps": "40", "horizontal_jitter_pps": "2", "vertical_jitter_pps": "0", "smoothness_percent": "80", "ramp_up_ms": "250", "max_step_px": "2", "acceleration_pps2": "120", "deceleration_pps2": "240", "motion_curve": "S-curve", "jitter_enabled": True},
    "Fast Response": {"motion_strength_pps": "60", "horizontal_jitter_pps": "2", "vertical_jitter_pps": "1", "smoothness_percent": "55", "ramp_up_ms": "100", "max_step_px": "2", "acceleration_pps2": "260", "deceleration_pps2": "400", "motion_curve": "Ease-in", "jitter_enabled": True},
    "Strong Shake": {"motion_strength_pps": "80", "horizontal_jitter_pps": "90", "vertical_jitter_pps": "70", "smoothness_percent": "18", "update_rate_hz": "240", "ramp_up_ms": "40", "jitter_rate_hz": "16", "jitter_randomness_percent": "20", "jitter_axis_phase_deg": "90", "jitter_waveform": "Random blend", "max_step_px": "10", "acceleration_pps2": "4000", "deceleration_pps2": "5000", "motion_curve": "Linear", "jitter_enabled": True},
    "Extreme": {"motion_strength_pps": "120", "horizontal_jitter_pps": "180", "vertical_jitter_pps": "150", "smoothness_percent": "5", "update_rate_hz": "360", "ramp_up_ms": "0", "jitter_rate_hz": "24", "jitter_randomness_percent": "40", "jitter_axis_phase_deg": "135", "jitter_waveform": "Square", "max_step_px": "18", "acceleration_pps2": "8000", "deceleration_pps2": "9000", "motion_curve": "Linear", "jitter_enabled": True},
}


@dataclass(frozen=True)
class MotionSettings:
    angle_deg: float = 90.0
    strength_pps: float = 80.0
    jitter_enabled: bool = True
    horizontal_jitter_pps: float = 55.0
    vertical_jitter_pps: float = 40.0
    smoothness: float = 25.0
    update_rate_hz: float = 240.0
    ramp_up_ms: float = 80.0
    jitter_rate_hz: float = 14.0
    jitter_randomness: float = 25.0
    jitter_axis_phase_deg: float = 90.0
    jitter_waveform: str = "Random blend"
    max_step_px: int = 8
    acceleration_pps2: float = 2500.0
    deceleration_pps2: float = 3500.0
    motion_curve: str = "S-curve"


_FIELD_KEYS = (
    ("angle_deg", "motion_angle_deg"),
    ("strength_pps", "motion_strength_pps"),
    ("horizontal_jitter_pps", "horizontal_jitter_pps"),
    ("vertical_jitter_pps", "vertical_jitter_pps"),
    ("smoothness", "smoothness_percent"),
    ("update_rate_hz", "update_rate_hz"),
    ("ramp_up_ms", "ramp_up_ms"),
    ("jitter_rate_hz", "jitter_rate_hz"),
    ("jitter_randomness", "jitter_randomness_percent"),
    ("jitter_axis_phase_deg", "jitter_axis_phase_deg"),
    ("max_step_px", "max_step_px"),
    ("acceleration_pps2", "acceleration_pps2"),
    ("deceleration_pps2", "deceleration_pps2"),
)


def _number(raw: Any, default: Any, key: str) -> float:
    try:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError
    except (TypeError, ValueError):
        value = float(default)
    low, high = MOTION_LIMITS[key]
    return max(low, min(high, value))


def _bool(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def motion_settings_from_mapping(raw: Mapping[str, Any] | None) -> MotionSettings:
    values = dict(MOTION_DEFAULTS)
    if raw:
        values.update(raw)
    converted = {}
    for field, key in _FIELD_KEYS:
        value = _number(values.get(key), MOTION_DEFAULTS[key], key)
        converted[field] = int(value) if key == "max_step_px" else value
    converted["jitter_enabled"] = _bool(values.get("jitter_enabled"), bool(MOTION_DEFAULTS["jitter_enabled"]))
    waveform = values.get("jitter_waveform")
    converted["jitter_waveform"] = waveform if waveform in JITTER_WAVEFORMS else MOTION_DEFAULTS["jitter_waveform"]
    curve = values.get("motion_curve")
    converted["motion_curve"] = curve if curve in MOTION_CURVES else MOTION_DEFAULTS["motion_curve"]
    return MotionSettings(**converted)


def _compact(value: float | int) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def motion_settings_to_mapping(settings: MotionSettings) -> dict[str, Any]:
    result = {key: _compact(getattr(settings, field)) for field, key in _FIELD_KEYS}
    result["jitter_enabled"] = settings.jitter_enabled
    result["jitter_waveform"] = settings.jitter_waveform
    result["motion_curve"] = settings.motion_curve
    return result


@dataclass
class SmoothMotionEngine:
    """Deterministic, stateful smooth motion generator."""

    velocity_x: float = 0.0
    velocity_y: float = 0.0
    residual_x: float = 0.0
    residual_y: float = 0.0
    filtered_x: float = 0.0
    filtered_y: float = 0.0
    jitter_phase: float = 0.0

    @staticmethod
    def _wave(phase: float, waveform: str, rng: random.Random) -> float:
        cycle = (phase / math.tau) % 1.0
        sine = math.sin(phase)
        if waveform == "Sine":
            return sine
        if waveform == "Triangle":
            return 1.0 - 4.0 * abs(round(cycle - 0.25) - (cycle - 0.25))
        if waveform == "Square":
            return 1.0 if sine >= 0.0 else -1.0
        # Random blend retains a coherent wave while adding deterministic noise.
        return 0.5 * sine + 0.5 * rng.uniform(-1.0, 1.0)

    @staticmethod
    def _ramp(progress: float, curve: str) -> float:
        progress = max(0.0, min(1.0, progress))
        if curve == "Ease-in":
            return progress * progress
        if curve == "S-curve":
            return progress * progress * (3.0 - 2.0 * progress)
        return progress

    def step(self, settings: MotionSettings, dt: float, elapsed: float,
             rng: random.Random = random) -> tuple[int, int]:
        dt = max(0.0, min(float(dt), 0.1))
        self.jitter_phase = (self.jitter_phase + math.tau * settings.jitter_rate_hz * dt) % math.tau
        jitter_x = jitter_y = 0.0
        if settings.jitter_enabled:
            randomness = max(0.0, min(100.0, settings.jitter_randomness)) / 100.0
            phase = self.jitter_phase
            axis_phase = math.radians(settings.jitter_axis_phase_deg)
            wave_x = self._wave(phase, settings.jitter_waveform, rng)
            wave_y = self._wave(phase + axis_phase, settings.jitter_waveform, rng)
            if settings.jitter_waveform != "Random blend":
                noise_x = rng.uniform(-1.0, 1.0)
                noise_y = rng.uniform(-1.0, 1.0)
                wave_x = wave_x * (1.0 - randomness) + noise_x * randomness
                wave_y = wave_y * (1.0 - randomness) + noise_y * randomness
            jitter_x = wave_x * settings.horizontal_jitter_pps
            jitter_y = wave_y * settings.vertical_jitter_pps

        angle = math.radians(settings.angle_deg)
        target_x = math.cos(angle) * settings.strength_pps + jitter_x
        target_y = math.sin(angle) * settings.strength_pps + jitter_y
        progress = 1.0 if settings.ramp_up_ms <= 0 else min(1.0, elapsed / (settings.ramp_up_ms / 1000.0))
        ramp = self._ramp(progress, settings.motion_curve)
        target_x *= ramp
        target_y *= ramp

        tau = (max(0.0, min(settings.smoothness, 100.0)) / 100.0) ** 2 * 0.250
        alpha = 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)
        self.filtered_x += (target_x - self.filtered_x) * alpha
        self.filtered_y += (target_y - self.filtered_y) * alpha

        target_speed = math.hypot(self.filtered_x, self.filtered_y)
        current_speed = math.hypot(self.velocity_x, self.velocity_y)
        limit = settings.acceleration_pps2 if target_speed >= current_speed else settings.deceleration_pps2
        max_delta = max(0.0, limit) * dt
        delta_x = self.filtered_x - self.velocity_x
        delta_y = self.filtered_y - self.velocity_y
        delta_len = math.hypot(delta_x, delta_y)
        if delta_len > max_delta > 0:
            scale = max_delta / delta_len
            delta_x *= scale
            delta_y *= scale
        elif max_delta <= 0:
            delta_x = delta_y = 0.0
        self.velocity_x += delta_x
        self.velocity_y += delta_y

        total_x = self.velocity_x * dt + self.residual_x
        total_y = self.velocity_y * dt + self.residual_y
        raw_x = int(total_x)
        raw_y = int(total_y)
        self.residual_x = total_x - raw_x
        self.residual_y = total_y - raw_y
        cap = max(1, int(settings.max_step_px))
        report_x = max(-cap, min(cap, raw_x))
        report_y = max(-cap, min(cap, raw_y))
        # A capped report must not leave hidden velocity that becomes a
        # multi-frame backlog when the target changes or stops.
        if dt > 0.0:
            if report_x != raw_x:
                self.velocity_x = report_x / dt
            if report_y != raw_y:
                self.velocity_y = report_y / dt
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
