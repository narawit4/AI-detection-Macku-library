import unittest
from dataclasses import replace

from motion import (
    JITTER_WAVEFORMS,
    MOTION_CURVES,
    MOTION_DEFAULTS,
    MOTION_PRESETS,
    MotionSettings,
    motion_settings_from_mapping,
    motion_settings_to_mapping,
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


if __name__ == "__main__":
    unittest.main()
