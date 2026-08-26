import unittest

from ai_targeting import AimSettings, TargetSnapshot
from combined_motion import CombinedMotionEngine, MotionSources
from motion import MotionSettings


class FixedJitter:
    def __init__(self, report):
        self.report = report
        self.calls = []

    def step(self, settings, dt, elapsed):
        self.calls.append((settings, dt, elapsed))
        return self.report


class FixedAim:
    def __init__(self, report):
        self.report = report
        self.calls = []

    def step(self, snapshot, settings, now):
        self.calls.append((snapshot, settings, now))
        return self.report


class SequenceJitter(FixedJitter):
    def __init__(self, *reports):
        super().__init__(reports[0])
        self.reports = iter(reports)

    def step(self, settings, dt, elapsed):
        self.calls.append((settings, dt, elapsed))
        return next(self.reports)


class CombinedMotionTests(unittest.TestCase):
    def test_combines_due_components_and_clamps_final_report(self):
        jitter = FixedJitter((100, -100))
        aim = FixedAim((60, -60))
        engine = CombinedMotionEngine(
            MotionSources(jitter=True, ai=True),
            jitter_engine_factory=lambda: jitter,
            aim_engine_factory=lambda: aim,
        )
        report = engine.step(
            MotionSettings(),
            TargetSnapshot(1, 1.0, "head", 200, 200),
            AimSettings(),
            dt=0.01,
            elapsed=0.02,
            now=1.01,
        )
        self.assertEqual(report, (127, -127))
        self.assertEqual(len(jitter.calls), 1)
        self.assertEqual(len(aim.calls), 1)

    def test_jitter_continues_when_ai_has_no_target(self):
        jitter = FixedJitter((2, -1))
        aim = FixedAim((0, 0))
        engine = CombinedMotionEngine(
            MotionSources(True, True),
            jitter_engine_factory=lambda: jitter,
            aim_engine_factory=lambda: aim,
        )
        self.assertEqual(
            engine.step(
                MotionSettings(), None, AimSettings(),
                dt=0.01, elapsed=0.02, now=1.0,
            ),
            (2, -1),
        )

    def test_disabled_component_factory_is_never_constructed(self):
        engine = CombinedMotionEngine(
            MotionSources(jitter=True, ai=False),
            jitter_engine_factory=lambda: FixedJitter((1, 1)),
            aim_engine_factory=lambda: self.fail("AI factory must stay unused"),
        )
        self.assertEqual(engine.poll_interval(MotionSettings(pulse_rate_hz=50)), 0.01)

    def test_ai_only_output_uses_ai_component(self):
        aim = FixedAim((-4, 7))
        engine = CombinedMotionEngine(
            MotionSources(jitter=False, ai=True),
            jitter_engine_factory=lambda: self.fail("Jitter factory must stay unused"),
            aim_engine_factory=lambda: aim,
        )
        self.assertEqual(
            engine.step(MotionSettings(), None, AimSettings(), dt=0.1, elapsed=1.0, now=2.0),
            (-4, 7),
        )

    def test_zero_components_produce_zero_report(self):
        engine = CombinedMotionEngine(
            MotionSources(True, True),
            jitter_engine_factory=lambda: FixedJitter((0, 0)),
            aim_engine_factory=lambda: FixedAim((0, 0)),
        )
        self.assertEqual(
            engine.step(MotionSettings(), None, AimSettings(), dt=0.1, elapsed=1.0, now=2.0),
            (0, 0),
        )

    def test_both_sources_false_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one motion source"):
            CombinedMotionEngine(MotionSources())

    def test_final_clamp_excess_is_discarded_on_next_zero_tick(self):
        jitter = SequenceJitter((200, -200), (0, 0))
        engine = CombinedMotionEngine(
            MotionSources(jitter=True), jitter_engine_factory=lambda: jitter
        )
        settings = MotionSettings()
        self.assertEqual(
            engine.step(settings, None, AimSettings(), dt=0.01, elapsed=0.0, now=0.0),
            (127, -127),
        )
        self.assertEqual(
            engine.step(settings, None, AimSettings(), dt=0.01, elapsed=0.01, now=0.01),
            (0, 0),
        )

    def test_ai_polling_takes_priority_over_jitter_cadence(self):
        engine = CombinedMotionEngine(
            MotionSources(jitter=True, ai=True),
            jitter_engine_factory=lambda: FixedJitter((0, 0)),
            aim_engine_factory=lambda: FixedAim((0, 0)),
        )
        self.assertEqual(engine.poll_interval(MotionSettings(pulse_rate_hz=20)), 1.0 / 240.0)


if __name__ == "__main__":
    unittest.main()
