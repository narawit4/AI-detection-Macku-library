import math
import unittest
from dataclasses import FrozenInstanceError

from ai_targeting import (
    AIM_LIMITS,
    AimSettings,
    Detection,
    DetectionFrameSnapshot,
    TargetSnapshot,
    AimMovementEngine,
    aim_settings_from_mapping,
    aim_settings_to_mapping,
    analyze_detections,
    detection_aim_point,
    select_target,
)


class AimSettingsTests(unittest.TestCase):
    def test_canonical_limits_match_approved_ui_and_config_contract(self):
        self.assertEqual(AIM_LIMITS, {
            "confidence": (0.05, 0.95),
            "aim_strength": (0.05, 2.0),
            "smoothing": (0.0, 0.95),
            "max_step": (1.0, 127.0),
        })

    def test_mapping_clamps_non_finite_and_out_of_range_values(self):
        settings = aim_settings_from_mapping({
            "confidence": "-4", "aim_strength": "-1",
            "smoothing": "1", "max_step": "999",
        })
        self.assertEqual(settings, AimSettings(
            confidence=0.05, aim_strength=0.05, smoothing=0.95, max_step=127,
        ))

    def test_mapping_rejects_invalid_values_using_defaults(self):
        settings = aim_settings_from_mapping({
            "confidence": "oops", "aim_strength": None,
            "smoothing": float("inf"), "max_step": "not-an-int",
        })
        self.assertEqual(settings, AimSettings())

    def test_mapping_rejects_boolean_numeric_values_using_defaults(self):
        settings = aim_settings_from_mapping({
            "confidence": True,
            "aim_strength": False,
            "smoothing": True,
            "max_step": False,
        })
        self.assertEqual(settings, AimSettings())

    def test_max_step_is_integral_and_limits_are_exposed(self):
        settings = aim_settings_from_mapping({"max_step": "20.9"})
        self.assertEqual(settings.max_step, 20)
        self.assertEqual(AIM_LIMITS["max_step"], (1.0, 127.0))

    def test_serialization_uses_compact_strings(self):
        settings = AimSettings(0.5, 2.0, 0.25, 20)
        self.assertEqual(aim_settings_to_mapping(settings), {
            "confidence": "0.5",
            "aim_strength": "2",
            "smoothing": "0.25",
            "max_step": "20",
        })


class TargetSelectionTests(unittest.TestCase):
    def test_analysis_filters_confidence_and_preserves_selected_box_index(self):
        low = Detection(1, 2, 10, 20, 0.20, 7)
        player = Detection(20, 30, 60, 130, 0.80, 0)
        head = Detection(140, 140, 180, 180, 0.90, 7)

        result = analyze_detections(
            (low, player, head),
            AimSettings(confidence=0.35),
            sequence=4,
            captured_at=10.0,
        )

        self.assertEqual(result.frame, DetectionFrameSnapshot(
            sequence=4,
            captured_at=10.0,
            detections=(player, head),
            selected_index=1,
        ))
        self.assertIsNotNone(result.target)
        self.assertEqual(result.target.target_class, "head")

    def test_detection_frame_is_deeply_immutable_and_empty_analysis_is_publishable(self):
        result = analyze_detections(
            (), AimSettings(), sequence=9, captured_at=20.0
        )
        self.assertIsNone(result.target)
        self.assertEqual(result.frame.detections, ())
        self.assertIsNone(result.frame.selected_index)
        with self.assertRaises(FrozenInstanceError):
            result.frame.sequence = 10

    def test_head_has_priority_over_nearer_player(self):
        detections = (
            Detection(150, 140, 170, 200, 0.90, 0),
            Detection(20, 20, 40, 40, 0.80, 7),
        )
        target = select_target(
            detections, AimSettings(), sequence=1, captured_at=10.0
        )
        self.assertEqual(target.target_class, "head")
        self.assertEqual((target.aim_x, target.aim_y), (30.0, 30.0))

    def test_player_fallback_aims_twenty_percent_below_top(self):
        target = select_target(
            (Detection(100, 50, 140, 150, 0.90, 0),),
            AimSettings(), sequence=2, captured_at=11.0,
        )
        self.assertEqual((target.aim_x, target.aim_y), (120.0, 70.0))

    def test_same_class_target_is_retained_within_association_radius(self):
        previous = TargetSnapshot(1, 10.0, "head", 40.0, 40.0)
        target = select_target(
            (
                Detection(42, 42, 52, 52, 0.9, 7),
                Detection(150, 150, 160, 160, 0.9, 7),
            ),
            AimSettings(), sequence=2, captured_at=10.01, previous=previous,
        )
        self.assertEqual((target.aim_x, target.aim_y), (47.0, 47.0))

    def test_rejects_below_confidence(self):
        self.assertIsNone(select_target(
            (Detection(0, 0, 10, 10, 0.34, 7),),
            AimSettings(), sequence=1, captured_at=1.0,
        ))

    def test_ignores_unknown_classes(self):
        self.assertIsNone(select_target(
            (Detection(0, 0, 10, 10, 1.0, 99),),
            AimSettings(), sequence=1, captured_at=1.0,
        ))

    def test_selects_nearest_candidate_to_crosshair(self):
        target = select_target(
            (
                Detection(0, 0, 10, 10, 0.9, 7),
                Detection(150, 150, 170, 170, 0.9, 7),
            ),
            AimSettings(), sequence=1, captured_at=1.0,
        )
        self.assertEqual((target.aim_x, target.aim_y), (160.0, 160.0))

    def test_switches_immediately_from_player_to_head(self):
        previous = TargetSnapshot(1, 1.0, "player", 25.0, 25.0)
        target = select_target(
            (Detection(5, 5, 15, 15, 0.9, 7), Detection(20, 20, 30, 30, 0.9, 0)),
            AimSettings(), sequence=2, captured_at=2.0, previous=previous,
        )
        self.assertEqual(target.target_class, "head")

    def test_empty_detections_return_none(self):
        self.assertIsNone(select_target((), AimSettings(), sequence=1, captured_at=1.0))


class DetectionAimPointTests(unittest.TestCase):
    def test_public_aim_point_preserves_head_and_player_contract(self):
        self.assertEqual(
            detection_aim_point(Detection(10, 20, 30, 40, 0.9, 7)),
            ("head", 20.0, 30.0),
        )
        self.assertEqual(
            detection_aim_point(Detection(10, 20, 30, 120, 0.9, 0)),
            ("player", 20.0, 40.0),
        )
        self.assertIsNone(
            detection_aim_point(Detection(10, 20, 30, 40, 0.9, 4))
        )


class AimMovementEngineTests(unittest.TestCase):
    def test_consumes_each_snapshot_only_once(self):
        engine = AimMovementEngine()
        target = TargetSnapshot(1, 10.0, "head", 200.0, 160.0)
        first = engine.step(target, AimSettings(smoothing=0.0), 10.01)
        second = engine.step(target, AimSettings(smoothing=0.0), 10.02)
        self.assertNotEqual(first, (0, 0))
        self.assertEqual(second, (0, 0))

    def test_stale_target_never_moves(self):
        engine = AimMovementEngine()
        target = TargetSnapshot(1, 10.0, "head", 200.0, 160.0)
        self.assertEqual(engine.step(target, AimSettings(), 10.151), (0, 0))

    def test_excess_is_clamped_and_not_queued(self):
        engine = AimMovementEngine()
        settings = AimSettings(aim_strength=2.0, smoothing=0.0, max_step=5)
        first = engine.step(
            TargetSnapshot(1, 1.0, "head", 320.0, 160.0), settings, 1.0
        )
        second = engine.step(
            TargetSnapshot(2, 1.01, "head", 160.0, 160.0), settings, 1.01
        )
        self.assertEqual(first, (5, 0))
        self.assertEqual(second, (0, 0))

    def test_dead_zone_produces_no_movement_and_resets_state(self):
        engine = AimMovementEngine()
        self.assertEqual(engine.step(TargetSnapshot(1, 1, "head", 166, 160), AimSettings(aim_strength=1, smoothing=0), 1), (6, 0))
        self.assertEqual(engine.step(TargetSnapshot(2, 1.01, "head", 161, 160), AimSettings(smoothing=0), 1.01), (0, 0))
        self.assertEqual(engine.step(TargetSnapshot(3, 1.02, "head", 164, 160), AimSettings(aim_strength=1, smoothing=0), 1.02), (4, 0))

    def test_strength_scales_error(self):
        engine = AimMovementEngine()
        self.assertEqual(engine.step(TargetSnapshot(1, 1, "head", 170, 160), AimSettings(aim_strength=0.5, smoothing=0), 1), (5, 0))

    def test_smoothing_interpolates_from_previous_axis(self):
        engine = AimMovementEngine()
        settings = AimSettings(aim_strength=1, smoothing=0.5, max_step=127)
        self.assertEqual(engine.step(TargetSnapshot(1, 1, "head", 170, 160), settings, 1), (5, 0))
        self.assertEqual(engine.step(TargetSnapshot(2, 1.01, "head", 170, 160), settings, 1.01), (7, 0))

    def test_acceleration_is_limited_per_axis(self):
        engine = AimMovementEngine()
        settings = AimSettings(aim_strength=1, smoothing=0, max_step=127)
        self.assertEqual(engine.step(TargetSnapshot(1, 1, "head", 200, 220), settings, 1), (6, 6))
        self.assertEqual(engine.step(TargetSnapshot(2, 1.01, "head", 300, 100), settings, 1.01), (12, 0))

    def test_negative_fractional_accumulation_is_symmetric(self):
        engine = AimMovementEngine()
        settings = AimSettings(aim_strength=0.15, smoothing=0, max_step=127)
        target = lambda seq: TargetSnapshot(seq, 1 + seq / 100, "head", 150, 160)
        self.assertEqual([engine.step(target(i), settings, 1 + i / 100) for i in range(1, 6)], [(-1, 0), (-2, 0), (-1, 0), (-2, 0), (-1, 0)])

    def test_reset_discards_motion_state(self):
        engine = AimMovementEngine()
        settings = AimSettings(aim_strength=1, smoothing=0.5, max_step=127)
        engine.step(TargetSnapshot(1, 1, "head", 200, 160), settings, 1)
        engine.reset()
        self.assertEqual(engine.step(TargetSnapshot(2, 1.01, "head", 200, 160), settings, 1.01), (6, 0))

    def test_future_timestamp_is_not_stale(self):
        engine = AimMovementEngine()
        self.assertEqual(engine.step(TargetSnapshot(1, 20, "head", 200, 160), AimSettings(smoothing=0), 10), (6, 0))

    def test_max_step_127_is_honored(self):
        engine = AimMovementEngine()
        settings = AimSettings(aim_strength=2, smoothing=0, max_step=127)
        reports = [engine.step(TargetSnapshot(i, 1 + i / 100, "head", 320, 320), settings, 1 + i / 100) for i in range(1, 40)]
        self.assertEqual(reports[-1], (127, 127))

    def test_none_target_resets_state(self):
        engine = AimMovementEngine()
        settings = AimSettings(aim_strength=1, smoothing=0.5, max_step=127)
        engine.step(TargetSnapshot(1, 1, "head", 200, 160), settings, 1)
        self.assertEqual(engine.step(None, settings, 1.01), (0, 0))
        self.assertEqual(engine.step(TargetSnapshot(2, 1.02, "head", 200, 160), settings, 1.02), (6, 0))


if __name__ == "__main__":
    unittest.main()
