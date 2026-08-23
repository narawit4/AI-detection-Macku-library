import unittest
import random
from dataclasses import replace

from motion import (
    JITTER_WAVEFORMS,
    MOTION_CURVES,
    MOTION_DEFAULTS,
    MOTION_PRESETS,
    MotionSettings,
    motion_settings_from_mapping,
    motion_settings_to_mapping,
    SmoothMotionEngine,
    TriggerGate,
)


class MotionSettingsTests(unittest.TestCase):
    def test_defaults_match_the_approved_strong_jitter_starting_point(self):
        settings = motion_settings_from_mapping({})
        self.assertEqual(settings.angle_deg, 90.0)
        self.assertEqual(settings.strength_pps, 80.0)
        self.assertEqual(settings.horizontal_jitter_pps, 55.0)
        self.assertEqual(settings.vertical_jitter_pps, 40.0)
        self.assertEqual(settings.update_rate_hz, 240.0)

    def test_numeric_values_are_clamped_and_invalid_choices_use_defaults(self):
        settings = motion_settings_from_mapping({
            "motion_angle_deg": -10,
            "motion_strength_pps": 900,
            "jitter_rate_hz": "bad",
            "jitter_waveform": "Saw",
            "motion_curve": "Instant",
        })
        self.assertEqual(settings.angle_deg, 0.0)
        self.assertEqual(settings.strength_pps, 500.0)
        self.assertEqual(settings.jitter_rate_hz, float(MOTION_DEFAULTS["jitter_rate_hz"]))
        self.assertIn(settings.jitter_waveform, JITTER_WAVEFORMS)
        self.assertIn(settings.motion_curve, MOTION_CURVES)

    def test_all_six_approved_presets_round_trip(self):
        self.assertEqual(
            tuple(MOTION_PRESETS),
            ("Ultra Stable", "Soft", "Balanced", "Fast Response", "Strong Shake", "Extreme"),
        )
        for name, raw in MOTION_PRESETS.items():
            settings = motion_settings_from_mapping(raw)
            restored = motion_settings_from_mapping(motion_settings_to_mapping(settings))
            self.assertEqual(restored, settings, name)

    def test_motion_settings_are_immutable(self):
        settings = MotionSettings()
        with self.assertRaises(AttributeError):
            settings.strength_pps = 20


class SmoothMotionEngineTests(unittest.TestCase):
    def test_angle_uses_screen_coordinates(self):
        engine = SmoothMotionEngine()
        settings = replace(MotionSettings(), angle_deg=90, strength_pps=100,
                           jitter_enabled=False, smoothness=0, ramp_up_ms=0,
                           acceleration_pps2=10000, max_step_px=50)
        x, y = engine.step(settings, 0.1, 1.0, random.Random(1))
        self.assertEqual(x, 0)
        self.assertGreater(y, 0)

    def test_fractional_motion_accumulates(self):
        engine = SmoothMotionEngine()
        settings = replace(MotionSettings(), angle_deg=0, strength_pps=3,
                           jitter_enabled=False, smoothness=0, ramp_up_ms=0,
                           acceleration_pps2=10000, max_step_px=50)
        reports = [engine.step(settings, 0.1, 1.0, random.Random(1))[0] for _ in range(10)]
        self.assertEqual(sum(reports), 3)

    def test_balanced_jitter_has_near_zero_net_drift(self):
        engine = SmoothMotionEngine()
        settings = replace(MotionSettings(), strength_pps=0, jitter_enabled=True,
                           horizontal_jitter_pps=20, vertical_jitter_pps=0,
                           jitter_rate_hz=1, jitter_randomness=0, jitter_waveform="Sine",
                           smoothness=0, ramp_up_ms=0, acceleration_pps2=10000,
                           max_step_px=50)
        reports = [engine.step(settings, 0.01, 1.0, random.Random(2))[0] for _ in range(100)]
        self.assertLessEqual(abs(sum(reports)), 1)

    def test_max_step_discards_excess_without_backlog(self):
        engine = SmoothMotionEngine()
        strong = replace(MotionSettings(), angle_deg=0, strength_pps=500,
                         jitter_enabled=False, smoothness=0, ramp_up_ms=0,
                         acceleration_pps2=10000, max_step_px=2)
        stopped = replace(strong, strength_pps=0)
        self.assertEqual(engine.step(strong, 0.1, 1.0, random.Random(3))[0], 2)
        self.assertEqual(engine.step(stopped, 0.1, 1.0, random.Random(3))[0], 0)


class TriggerGateTests(unittest.TestCase):
    def test_modifier_is_required_when_configured(self):
        gate = TriggerGate(trigger="Left", modifier="Right")
        gate.update_button("Left", True)
        self.assertFalse(gate.active)
        gate.update_button("Right", True)
        self.assertTrue(gate.active)
        gate.update_button("Right", False)
        self.assertFalse(gate.active)

    def test_reconfigure_and_clear_drop_held_state(self):
        gate = TriggerGate(trigger="Left", modifier="None")
        gate.update_button("Left", True)
        self.assertTrue(gate.active)
        gate.configure("Mouse4", "None")
        self.assertFalse(gate.active)
        gate.update_button("Mouse4", True)
        gate.clear()
        self.assertFalse(gate.active)


if __name__ == "__main__":
    unittest.main()
