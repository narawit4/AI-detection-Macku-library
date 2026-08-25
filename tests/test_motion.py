from dataclasses import FrozenInstanceError
import unittest

from motion import (
    MOTION_PRESETS,
    MotionSettings,
    PairedPulseEngine,
    motion_settings_from_mapping,
    motion_settings_to_mapping,
    TriggerGate,
)


class MotionSettingsTests(unittest.TestCase):
    def test_defaults_match_balanced_paired_pulse(self):
        self.assertEqual(MotionSettings(), MotionSettings(2.0, 30.0, "Smooth"))
        self.assertEqual(
            motion_settings_to_mapping(MotionSettings()),
            {"pulse_size_px": "2", "pulse_rate_hz": "30", "ramp_mode": "Smooth"},
        )

    def test_motion_settings_snapshot_is_immutable(self):
        settings = MotionSettings()
        with self.assertRaises(FrozenInstanceError):
            settings.pulse_size_px = 4.0

    def test_values_are_clamped_and_invalid_ramp_uses_default(self):
        settings = motion_settings_from_mapping({
            "pulse_size_px": "99",
            "pulse_rate_hz": "-4",
            "ramp_mode": "Unknown",
        })
        self.assertEqual(settings, MotionSettings(8.0, 10.0, "Smooth"))

    def test_huge_integer_values_use_safe_defaults_instead_of_overflowing(self):
        huge_integer = 10**10000
        settings = motion_settings_from_mapping({
            "pulse_size_px": huge_integer,
            "pulse_rate_hz": huge_integer,
        })
        self.assertEqual(settings, MotionSettings())

    def test_presets_are_exact_and_round_trip(self):
        self.assertEqual(tuple(MOTION_PRESETS), ("Soft", "Balanced", "Strong"))
        expected = {
            "Soft": MotionSettings(1.0, 20.0, "Smooth"),
            "Balanced": MotionSettings(2.0, 30.0, "Smooth"),
            "Strong": MotionSettings(4.0, 45.0, "Instant"),
        }
        for name, want in expected.items():
            got = motion_settings_from_mapping(MOTION_PRESETS[name])
            self.assertEqual(got, want)
            self.assertEqual(
                motion_settings_from_mapping(motion_settings_to_mapping(got)),
                want,
            )


class PairedPulseEngineTests(unittest.TestCase):
    def test_complete_pairs_alternate_order_and_have_zero_net_motion(self):
        engine = PairedPulseEngine()
        settings = MotionSettings(2.0, 10.0, "Instant")
        reports = [engine.step(settings, 0.05, elapsed) for elapsed in (0.0, 0.05, 0.10, 0.15)]
        self.assertEqual(reports, [(0, -2), (0, 2), (0, 2), (0, -2)])
        self.assertEqual(tuple(map(sum, zip(*reports))), (0, 0))

    def test_many_complete_pairs_never_emit_horizontal_or_net_drift(self):
        engine = PairedPulseEngine()
        settings = MotionSettings(3.0, 20.0, "Instant")
        reports = [engine.step(settings, 0.025, index * 0.025) for index in range(400)]
        self.assertTrue(all(x == 0 for x, _y in reports))
        self.assertEqual(sum(y for _x, y in reports), 0)

    def test_smooth_ramp_has_exact_early_fractional_residual_sequence(self):
        engine = PairedPulseEngine()
        settings = MotionSettings(2.0, 30.0, "Smooth")
        reports = [engine.step(settings, 1 / 60, index / 60) for index in range(10)]
        self.assertEqual(
            reports,
            [
                (0, 0), (0, 0),
                (0, 0), (0, 0),
                (0, -1), (0, 1),
                (0, 1), (0, -1),
                (0, -2), (0, 2),
            ],
        )

    def test_late_step_discards_missed_half_pulses(self):
        engine = PairedPulseEngine()
        settings = MotionSettings(2.0, 10.0, "Instant")
        self.assertEqual(engine.step(settings, 0.05, 0.0), (0, -2))
        self.assertEqual(engine.step(settings, 0.1, 1.0), (0, 2))

    def test_reset_starts_a_fresh_up_down_pair(self):
        engine = PairedPulseEngine()
        settings = MotionSettings(2.0, 30.0, "Instant")
        engine.step(settings, 1 / 60, 0.0)
        engine.reset()
        self.assertEqual(engine.step(settings, 1 / 60, 0.0), (0, -2))


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
