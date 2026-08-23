"""Motion settings domain model and serialization helpers."""

from dataclasses import dataclass
import math
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
