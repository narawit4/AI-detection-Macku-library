"""Pure composition of Jitter and AI Aim movement."""

from collections.abc import Callable
from dataclasses import dataclass

from ai_targeting import AimMovementEngine, AimSettings, TargetSnapshot
from motion import MotionSettings, PairedPulseEngine


@dataclass(frozen=True)
class MotionSources:
    jitter: bool = False
    ai: bool = False

    @property
    def any(self) -> bool:
        return self.jitter or self.ai


class CombinedMotionEngine:
    def __init__(
        self,
        sources: MotionSources,
        jitter_engine_factory: Callable[[], object] = PairedPulseEngine,
        aim_engine_factory: Callable[[], object] = AimMovementEngine,
    ) -> None:
        if not sources.any:
            raise ValueError("At least one motion source must be selected")
        self.sources = sources
        self._jitter = jitter_engine_factory() if sources.jitter else None
        self._aim = aim_engine_factory() if sources.ai else None

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
        jitter = (
            self._jitter.step(motion_settings, dt, elapsed)
            if self._jitter is not None else (0, 0)
        )
        aim = (
            self._aim.step(target, aim_settings, now)
            if self._aim is not None else (0, 0)
        )
        return (
            max(-127, min(127, int(jitter[0]) + int(aim[0]))),
            max(-127, min(127, int(jitter[1]) + int(aim[1]))),
        )

    def poll_interval(self, motion_settings: MotionSettings) -> float:
        if self.sources.ai:
            return 1.0 / 240.0
        rate = max(20.0, min(120.0, float(motion_settings.pulse_rate_hz)))
        return 1.0 / (rate * 2.0)
