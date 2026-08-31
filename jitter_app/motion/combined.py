"""Pure composition of Jitter and AI Aim movement."""

from collections.abc import Callable
from dataclasses import dataclass
import math

from jitter_app.ai.targeting import AimMovementEngine, AimSettings, TargetSnapshot
from .engine import DEFAULT_SERVO_HZ, MotionSettings, PairedPulseEngine


@dataclass(frozen=True)
class MotionSources:
    jitter: bool = False
    ai: bool = False

    @property
    def any(self) -> bool:
        return self.jitter or self.ai


def compose_motion_components(
    jitter: tuple[int, int],
    aim: tuple[int, int],
) -> tuple[int, int]:
    """Compose one Jitter and AI report without retaining excess movement."""
    return (
        max(-127, min(127, int(jitter[0]) + int(aim[0]))),
        max(-127, min(127, int(jitter[1]) + int(aim[1]))),
    )


class CombinedMotionEngine:
    def __init__(
        self,
        sources: MotionSources,
        jitter_engine_factory: Callable[[], object] = PairedPulseEngine,
        aim_engine_factory: Callable[[], object] | None = None,
        ai_poll_hz: float = DEFAULT_SERVO_HZ,
    ) -> None:
        if not sources.any:
            raise ValueError("At least one motion source must be selected")
        try:
            self._servo_hz = float(ai_poll_hz)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("AI poll rate must be positive and finite") from exc
        if not math.isfinite(self._servo_hz) or self._servo_hz <= 0.0:
            raise ValueError("AI poll rate must be positive and finite")
        self.sources = sources
        self._jitter = jitter_engine_factory() if sources.jitter else None
        self._aim = None
        if sources.ai:
            self._aim = (
                aim_engine_factory()
                if aim_engine_factory is not None
                else AimMovementEngine(nominal_hz=self._servo_hz)
            )

    def step(
        self,
        motion_settings: MotionSettings,
        target: TargetSnapshot | None,
        aim_settings: AimSettings,
        *,
        dt: float,
        elapsed: float,
        now: float,
    ) -> tuple[int, int]:
        jitter, aim = self.step_components(
            motion_settings,
            target,
            aim_settings,
            dt=dt,
            elapsed=elapsed,
            now=now,
        )
        return compose_motion_components(jitter, aim)

    def step_components(
        self,
        motion_settings: MotionSettings,
        target: TargetSnapshot | None,
        aim_settings: AimSettings,
        *,
        dt: float,
        elapsed: float,
        now: float,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Advance both sources once and retain their independent reports."""
        jitter = (
            self._jitter.step(motion_settings, dt, elapsed)
            if self._jitter is not None else (0, 0)
        )
        aim = (
            self._aim.step(target, aim_settings, now)
            if self._aim is not None else (0, 0)
        )
        return jitter, aim

    def poll_interval(self, _motion_settings: MotionSettings) -> float:
        return 1.0 / self._servo_hz
