import math
import unittest
from dataclasses import FrozenInstanceError

from jitter_app.ai.targeting import (
    AIM_LIMITS,
    AimSettings,
    DEFAULT_RESPONSE_CURVE,
    TARGET_AREAS,
    Detection,
    DetectionFrameSnapshot,
    RESPONSE_CURVE_X,
    TargetSnapshot,
    AimMovementEngine,
    aim_settings_from_mapping,
    aim_settings_to_mapping,
    analyze_detections,
    detection_aim_point,
    response_curve_value,
    select_target,
    validated_response_curve,
    validated_target_area,
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
            "response_curve": ["0", "0.12", "0.35", "0.68", "1"],
        })

    def test_target_area_defaults_validates_and_round_trips(self):
        self.assertEqual(TARGET_AREAS, ("head", "upper_body", "chest"))
        self.assertEqual(AimSettings().target_area, "head")
        self.assertEqual(validated_target_area("upper_body"), "upper_body")
        for raw in (None, "Head", "body", 1, True):
            with self.subTest(raw=raw):
                self.assertEqual(validated_target_area(raw), "head")
        settings = aim_settings_from_mapping({"target_area": "chest"})
        self.assertEqual(settings.target_area, "chest")
        self.assertNotIn("target_area", aim_settings_to_mapping(settings))

    def test_response_curve_defaults_and_round_trips(self):
        settings = aim_settings_from_mapping({
            "response_curve": ["0", "0.1", "0.3", "0.7", "0.9"],
        })
        self.assertEqual(settings.response_curve, (0.0, 0.1, 0.3, 0.7, 0.9))
        self.assertEqual(
            aim_settings_to_mapping(settings)["response_curve"],
            ["0", "0.1", "0.3", "0.7", "0.9"],
        )

    def test_malformed_curve_uses_complete_default(self):
        invalid = (
            None,
            [0, .1],
            [0, .4, .3, .8, 1],
            [.1, .2, .3, .4, .5],
            [0, .2, .3, .4, float("nan")],
            [0, .2, .3, .4, float("inf")],
            [0, .2, True, .4, 1],
            [0, .2, .3, .4, 1.1],
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                self.assertEqual(validated_response_curve(raw), DEFAULT_RESPONSE_CURVE)

    def test_flat_curve_segment_is_valid(self):
        self.assertEqual(
            validated_response_curve([0, .2, .2, .7, 1]),
            (0.0, 0.2, 0.2, 0.7, 1.0),
        )

    def test_unordered_curve_iterable_uses_complete_default(self):
        unordered = frozenset((0.0, 0.08, 0.18, 0.30, 0.33))

        self.assertEqual(
            validated_response_curve(unordered),
            DEFAULT_RESPONSE_CURVE,
        )

    def test_flat_curve_segment_stays_at_its_knot_value(self):
        curve = (0.0, 0.9, 0.9, 0.9, 1.0)
        samples = [
            response_curve_value(curve, distance)
            for distance in (0.25, 0.2509, 0.5, 0.7499)
        ]

        self.assertEqual(samples, [0.9, 0.9, 0.9, 0.9])
        self.assertEqual(samples, sorted(samples))

    def test_aim_settings_is_immutable_with_response_curve(self):
        settings = AimSettings(response_curve=(0, .2, .4, .7, 1))
        with self.assertRaises(FrozenInstanceError):
            settings.response_curve = DEFAULT_RESPONSE_CURVE

    def test_monotone_curve_hits_points_and_never_overshoots(self):
        curve = (0.0, 0.12, 0.35, 0.68, 1.0)
        for x, y in zip(RESPONSE_CURVE_X, curve):
            self.assertAlmostEqual(response_curve_value(curve, x), y)
        samples = [response_curve_value(curve, index / 100) for index in range(101)]
        self.assertEqual(samples, sorted(samples))
        self.assertEqual(response_curve_value(curve, -1), 0.0)
        self.assertEqual(response_curve_value(curve, 2), 1.0)


class TargetSelectionTests(unittest.TestCase):
    def test_full_hd_analysis_selects_nearest_to_actual_frame_center(self):
        result = analyze_detections(
            (
                Detection(930, 500, 990, 580, 0.9, 7),
                Detection(930, 920, 990, 980, 0.9, 7),
            ),
            AimSettings(),
            sequence=8,
            captured_at=10.0,
            frame_width=1920,
            frame_height=1080,
        )
        self.assertEqual((result.target.aim_x, result.target.aim_y), (960.0, 540.0))
        self.assertEqual(
            (result.target.frame_width, result.target.frame_height),
            (1920, 1080),
        )
        self.assertEqual(
            (result.frame.frame_width, result.frame.frame_height),
            (1920, 1080),
        )
        self.assertEqual(result.frame.selected_index, 0)

    def test_native_exact_distance_tie_preserves_detector_order(self):
        result = analyze_detections(
            (
                Detection(900, 510, 940, 570, 0.9, 7),
                Detection(980, 510, 1020, 570, 0.9, 7),
            ),
            AimSettings(),
            sequence=9,
            captured_at=11.0,
            frame_width=1920,
            frame_height=1080,
        )
        self.assertEqual(result.frame.selected_index, 0)

    def test_analysis_rejects_invalid_frame_dimensions(self):
        for width, height in ((0, 1080), (1920, 0), (True, 1080), (1920, 1.5)):
            with self.subTest(width=width, height=height):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    analyze_detections(
                        (),
                        AimSettings(),
                        sequence=1,
                        captured_at=1.0,
                        frame_width=width,
                        frame_height=height,
                    )

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

    def test_nearest_player_beats_farther_head_in_current_frame(self):
        detections = (
            Detection(150, 140, 170, 200, 0.90, 0),
            Detection(20, 20, 40, 40, 0.80, 7),
        )
        target = select_target(
            detections, AimSettings(), sequence=1, captured_at=10.0
        )
        self.assertEqual(target.target_class, "player")
        self.assertEqual((target.aim_x, target.aim_y), (160.0, 152.0))

    def test_player_fallback_aims_twenty_percent_below_top(self):
        target = select_target(
            (Detection(100, 50, 140, 150, 0.90, 0),),
            AimSettings(), sequence=2, captured_at=11.0,
        )
        self.assertEqual((target.aim_x, target.aim_y), (120.0, 70.0))

    def test_upper_body_uses_player_box_and_keeps_head_in_overlay(self):
        head = Detection(150, 150, 170, 170, 0.95, 7)
        player = Detection(100, 50, 140, 150, 0.90, 0)

        result = analyze_detections(
            (head, player),
            AimSettings(target_area="upper_body"),
            sequence=3,
            captured_at=12.0,
        )

        self.assertEqual(result.frame.detections, (head, player))
        self.assertEqual(result.frame.selected_index, 1)
        self.assertEqual(result.target.target_class, "player")
        self.assertEqual((result.target.aim_x, result.target.aim_y), (120.0, 80.0))

    def test_chest_uses_player_box_and_never_falls_back_to_head(self):
        head = Detection(150, 150, 170, 170, 0.95, 7)
        self.assertIsNone(select_target(
            (head,),
            AimSettings(target_area="chest"),
            sequence=3,
            captured_at=12.0,
        ))
        target = select_target(
            (Detection(100, 50, 140, 150, 0.90, 0),),
            AimSettings(target_area="chest"),
            sequence=4,
            captured_at=13.0,
        )
        self.assertEqual((target.aim_x, target.aim_y), (120.0, 92.0))

    def test_current_frame_nearest_ignores_previous_target_association(self):
        previous = TargetSnapshot(1, 10.0, "head", 40.0, 40.0)
        target = select_target(
            (
                Detection(42, 42, 52, 52, 0.9, 7),
                Detection(150, 150, 160, 160, 0.9, 7),
            ),
            AimSettings(), sequence=2, captured_at=10.01, previous=previous,
        )
        self.assertEqual((target.aim_x, target.aim_y), (155.0, 155.0))

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

    def test_equal_distance_cross_class_tie_preserves_detector_order(self):
        player = Detection(130, 140, 150, 240, 0.9, 0)
        head = Detection(175, 155, 185, 165, 0.9, 7)

        result = analyze_detections(
            (player, head),
            AimSettings(),
            sequence=1,
            captured_at=1.0,
        )

        self.assertEqual(result.frame.selected_index, 0)
        self.assertEqual(result.target.target_class, "player")
        self.assertEqual((result.target.aim_x, result.target.aim_y), (140.0, 160.0))

    def test_switches_immediately_from_player_to_head(self):
        previous = TargetSnapshot(1, 1.0, "player", 25.0, 25.0)
        target = select_target(
            (Detection(155, 155, 165, 165, 0.9, 7), Detection(20, 20, 30, 30, 0.9, 0)),
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

    def test_body_aim_points_require_player_boxes(self):
        head = Detection(10, 20, 30, 40, 0.9, 7)
        player = Detection(10, 20, 30, 120, 0.9, 0)
        self.assertIsNone(detection_aim_point(head, "upper_body"))
        self.assertEqual(
            detection_aim_point(player, "upper_body"),
            ("player", 20.0, 50.0),
        )
        self.assertEqual(
            detection_aim_point(player, "chest"),
            ("player", 20.0, 62.0),
        )


class AimMovementEngineTests(unittest.TestCase):
    LINEAR_CURVE = (0.0, 0.25, 0.5, 0.75, 1.0)

    def test_geometry_change_resets_old_velocity_fraction_and_error(self):
        engine = AimMovementEngine(nominal_hz=1000.0)
        settings = AimSettings(smoothing=0.0, aim_strength=0.35, max_step=20)
        old = TargetSnapshot(1, 10.0, "head", 320.0, 320.0, 320, 320)
        for tick in range(20):
            engine.step(old, settings, 10.0 + tick / 1000.0)

        replacement = TargetSnapshot(
            2, 10.020, "head", 960.0, 440.0, 1920, 1080
        )
        actual = [
            engine.step(replacement, settings, 10.020 + tick / 1000.0)
            for tick in range(10)
        ]
        clean = AimMovementEngine(nominal_hz=1000.0)
        expected = [
            clean.step(replacement, settings, 10.020 + tick / 1000.0)
            for tick in range(10)
        ]
        self.assertEqual(actual, expected)

    def test_invalid_target_geometry_resets_and_emits_zero(self):
        engine = AimMovementEngine(nominal_hz=1000.0)
        settings = AimSettings(smoothing=0.0)
        engine.step(TargetSnapshot(1, 10.0, "head", 200, 160), settings, 10.0)

        self.assertEqual(
            engine.step(
                TargetSnapshot(2, 10.001, "head", 10, 10, 0, 1080),
                settings,
                10.001,
            ),
            (0, 0),
        )
        self.assertLessEqual(
            engine.step(
                TargetSnapshot(3, 10.002, "head", 154, 160),
                settings,
                10.002,
            )[0],
            0,
        )

    def test_response_curve_uses_native_half_radius_and_corner(self):
        engine = AimMovementEngine(nominal_hz=60.0)
        settings = AimSettings(
            smoothing=0.0,
            aim_strength=0.05,
            max_step=127,
            response_curve=(0.0, 0.0, 0.0, 0.0, 1.0),
        )
        radius = math.hypot(960.0, 540.0)
        half_radius = TargetSnapshot(
            1, 10.0, "head", 960.0 + radius / 2.0, 540.0, 1920, 1080
        )
        self.assertEqual(engine.step(half_radius, settings, 10.0), (0, 0))

        engine.reset()
        corner = TargetSnapshot(
            2, 10.1, "head", 1920.0, 1080.0, 1920, 1080
        )
        dx, dy = engine.step(corner, settings, 10.1)
        self.assertGreater(dx, 0)
        self.assertGreater(dy, 0)

    def test_same_normalized_distance_scales_output_by_each_frame_radius(self):
        settings = AimSettings(
            smoothing=0.0,
            aim_strength=0.01,
            max_step=127,
            response_curve=(0.0, 0.25, 0.50, 0.75, 1.0),
        )
        cases = (
            (320, 320, 10.0, 1),
            (1920, 1080, 20.0, 5),
        )
        for width, height, now, expected_dx in cases:
            with self.subTest(size=(width, height)):
                center_x = width / 2.0
                center_y = height / 2.0
                radius = math.hypot(center_x, center_y)
                target = TargetSnapshot(
                    1,
                    now,
                    "head",
                    center_x + radius / 2.0,
                    center_y,
                    width,
                    height,
                )
                engine = AimMovementEngine(nominal_hz=60.0)
                self.assertEqual(
                    engine.step(target, settings, now),
                    (expected_dx, 0),
                )

    def test_one_fresh_snapshot_produces_multiple_microsteps(self):
        engine = AimMovementEngine(nominal_hz=240)
        target = TargetSnapshot(1, 10.0, "head", 210.0, 160.0)
        reports = [
            engine.step(target, AimSettings(smoothing=0.0), 10.0 + index / 240)
            for index in range(8)
        ]
        self.assertGreater(sum(report[0] != 0 for report in reports), 1)
        self.assertTrue(all(report[1] == 0 for report in reports))

    def test_elapsed_time_not_tick_count_controls_total_displacement(self):
        totals = []
        for hz in (120, 288, 480, 1000):
            engine = AimMovementEngine(nominal_hz=hz)
            target = TargetSnapshot(1, 20.0, "head", 220.0, 160.0)
            reports = [
                engine.step(target, AimSettings(smoothing=0.0), 20 + index / hz)
                for index in range(int(hz * 0.1))
            ]
            totals.append(sum(x for x, _ in reports))
        self.assertLessEqual(max(totals) - min(totals), 2)

    def test_fresh_frames_keep_only_direction_compatible_fractional_carry(self):
        settings = AimSettings(
            aim_strength=0.05,
            smoothing=0.0,
            max_step=127,
            response_curve=self.LINEAR_CURVE,
        )
        same_direction = AimMovementEngine(nominal_hz=1000)
        same_direction_reports = []
        for index in range(20):
            frame_index = index // 5
            now = 10.0 + index / 1000
            same_direction_reports.append(
                same_direction.step(
                    TargetSnapshot(
                        frame_index + 1,
                        10.0 + frame_index / 200,
                        "head",
                        180.0,
                        160.0,
                    ),
                    settings,
                    now,
                )
            )

        reversed_direction = AimMovementEngine(nominal_hz=10)
        reverse_reports = [
            reversed_direction.step(
                TargetSnapshot(1, 10.0, "head", 163.0, 160.0),
                settings,
                10.0,
            ),
            reversed_direction.step(
                TargetSnapshot(2, 10.1, "head", 156.0, 160.0),
                settings,
                10.1,
            ),
        ]

        self.assertEqual(
            {
                "same_direction_total": sum(
                    report_x for report_x, _report_y in same_direction_reports
                ),
                "reverse_reports": reverse_reports,
            },
            {
                "same_direction_total": 1,
                "reverse_reports": [(0, 0), (-1, 0)],
            },
        )

    def test_smoothed_reversal_cannot_regenerate_incompatible_fractional_carry(self):
        settings = AimSettings(aim_strength=1.0, smoothing=0.65)
        cases = (
            (
                "positive x",
                (170.0, 160.0),
                (150.0, 160.0),
                [(0, 0), (1, 0), (0, 0), (0, 0)],
            ),
            (
                "negative x",
                (150.0, 160.0),
                (170.0, 160.0),
                [(0, 0), (-1, 0), (0, 0), (0, 0)],
            ),
            (
                "positive y",
                (160.0, 170.0),
                (160.0, 150.0),
                [(0, 0), (0, 1), (0, 0), (0, 0)],
            ),
            (
                "negative y",
                (160.0, 150.0),
                (160.0, 170.0),
                [(0, 0), (0, -1), (0, 0), (0, 0)],
            ),
        )

        for name, initial_point, reverse_point, expected in cases:
            with self.subTest(name=name):
                engine = AimMovementEngine(nominal_hz=60)
                reports = [
                    engine.step(
                        TargetSnapshot(
                            sequence,
                            10.0 if sequence == 1 else 10.0 + index / 60,
                            "head",
                            *point,
                        ),
                        settings,
                        10.0 + index / 60,
                    )
                    for index, (sequence, point) in enumerate((
                        (1, initial_point),
                        (1, initial_point),
                        (2, reverse_point),
                        (3, initial_point),
                    ))
                ]

                self.assertEqual(reports, expected)

    def test_zero_axis_cannot_regenerate_carry_while_other_axis_is_active(self):
        engine = AimMovementEngine(nominal_hz=60)
        settings = AimSettings(aim_strength=0.35, smoothing=0.2)

        reports = [
            engine.step(
                TargetSnapshot(sequence, 10.0 + index / 60, "head", aim_x, aim_y),
                settings,
                10.0 + index / 60,
            )
            for index, (sequence, aim_x, aim_y) in enumerate((
                (1, 170.0, 160.0),
                (2, 160.0, 165.0),
                (3, 170.0, 165.0),
            ))
        ]

        self.assertEqual(reports, [(0, 0), (0, 0), (0, 0)])

    def test_first_tick_uses_exact_nominal_interval(self):
        engine = AimMovementEngine(nominal_hz=60)
        settings = AimSettings(
            aim_strength=1.0,
            smoothing=0.0,
            max_step=127,
            response_curve=self.LINEAR_CURVE,
        )
        target = TargetSnapshot(1, 1_000.0, "head", 176.0, 160.0)

        self.assertEqual(engine.step(target, settings, 1_000.0), (6, 0))

    def test_fresh_sequence_replaces_remaining_error_and_old_excess(self):
        engine = AimMovementEngine(nominal_hz=10)
        settings = AimSettings(
            aim_strength=1.0,
            smoothing=0.0,
            max_step=5,
            response_curve=self.LINEAR_CURVE,
        )
        engine.step(TargetSnapshot(1, 100.0, "head", 320.0, 160.0), settings, 1.0)
        replacement = TargetSnapshot(2, 100.0, "head", 158.0, 160.0)

        self.assertEqual(engine.step(replacement, settings, 1.1), (-2, 0))
        self.assertEqual(engine.step(replacement, settings, 1.2), (0, 0))

    def test_smoothing_zero_is_immediate(self):
        target = TargetSnapshot(1, 10.0, "head", 176.0, 160.0)
        common = dict(
            aim_strength=1.0,
            max_step=127,
            response_curve=self.LINEAR_CURVE,
        )

        immediate = AimMovementEngine(nominal_hz=60).step(
            target, AimSettings(smoothing=0.0, **common), 10.0
        )

        self.assertEqual(immediate, (6, 0))

    def test_servo_constants_match_exact_acceleration_and_maximum_tau_contract(self):
        self.assertEqual(
            (
                AimMovementEngine.MAX_ACCELERATION,
                AimMovementEngine.MAX_SMOOTHING_TAU_S,
            ),
            (21_600.0, 0.200),
        )

    def test_point_ninety_five_smoothing_uses_exact_tau_over_multiple_ticks(self):
        engine = AimMovementEngine(nominal_hz=60)
        settings = AimSettings(
            aim_strength=1.0,
            smoothing=0.95,
            max_step=5,
            response_curve=self.LINEAR_CURVE,
        )
        target = TargetSnapshot(1, 100.0, "head", 320.0, 160.0)

        reports = [
            engine.step(target, settings, 10.0 + index / 60)
            for index in range(12)
        ]

        self.assertEqual(
            reports,
            [
                (0, 0), (1, 0), (1, 0), (1, 0),
                (2, 0), (2, 0), (2, 0), (3, 0),
                (2, 0), (3, 0), (3, 0), (3, 0),
            ],
        )

    def test_acceleration_uses_exact_21600_vector_limit_over_multiple_ticks(self):
        engine = AimMovementEngine(nominal_hz=60)
        settings = AimSettings(aim_strength=2.0, smoothing=0.0, max_step=127)
        target = TargetSnapshot(1, 100.0, "head", 320.0, 320.0)

        reports = [
            engine.step(target, settings, 10.0 + index / 60)
            for index in range(5)
        ]

        self.assertEqual(
            reports,
            [(4, 4), (8, 8), (13, 13), (17, 17), (21, 21)],
        )

    def test_each_report_is_clamped_to_max_step(self):
        engine = AimMovementEngine(nominal_hz=10)
        settings = AimSettings(aim_strength=2.0, smoothing=0.0, max_step=5)
        target = TargetSnapshot(1, 1.0, "head", 320.0, 160.0)

        self.assertEqual(engine.step(target, settings, 1.0), (5, 0))
        self.assertEqual(engine.step(target, settings, 1.1), (5, 0))

    def test_fractional_carry_accumulates_symmetrically(self):
        settings = AimSettings(
            aim_strength=0.05,
            smoothing=0.0,
            max_step=20,
            response_curve=self.LINEAR_CURVE,
        )
        reports = []
        for aim_x in (180.0, 140.0):
            engine = AimMovementEngine(nominal_hz=64)
            target = TargetSnapshot(1, 10.0, "head", aim_x, 160.0)
            reports.append([
                engine.step(target, settings, 10.0 + index / 64)
                for index in range(4)
            ])

        self.assertEqual(reports[0], [(0, 0), (1, 0), (1, 0), (1, 0)])
        self.assertEqual(reports[1], [(0, 0), (-1, 0), (-1, 0), (-1, 0)])

    def test_dead_zone_resets_all_motion_state(self):
        engine = AimMovementEngine(nominal_hz=60)
        settings = AimSettings(
            aim_strength=1.0,
            smoothing=0.95,
            max_step=20,
            response_curve=self.LINEAR_CURVE,
        )
        engine.step(TargetSnapshot(1, 10.0, "head", 320.0, 160.0), settings, 10.0)

        self.assertEqual(
            engine.step(TargetSnapshot(2, 10.1, "head", 161.0, 160.0), settings, 10.1),
            (0, 0),
        )
        self.assertEqual(
            engine.step(TargetSnapshot(3, 10.2, "head", 176.0, 160.0), settings, 10.2),
            (1, 0),
        )

    def test_nonzero_target_is_fresh_at_exact_deadline_and_stale_epsilon_after(self):
        settings = AimSettings(aim_strength=2.0, smoothing=0.0, max_step=20)
        target = TargetSnapshot(1, 10.0, "head", 320.0, 160.0)

        self.assertNotEqual(
            AimMovementEngine(nominal_hz=10).step(target, settings, 10.15),
            (0, 0),
        )
        self.assertEqual(
            AimMovementEngine(nominal_hz=10).step(
                target, settings, math.nextafter(10.15, math.inf)
            ),
            (0, 0),
        )

    def test_settled_sequence_cannot_reload_and_next_same_direction_uses_nominal_dt(self):
        engine = AimMovementEngine(nominal_hz=60)
        settings = AimSettings(
            aim_strength=1.0,
            smoothing=0.0,
            max_step=127,
            response_curve=self.LINEAR_CURVE,
        )
        settled = TargetSnapshot(1, 100.0, "head", 166.0, 160.0)

        self.assertEqual(engine.step(settled, settings, 10.0), (6, 0))
        self.assertEqual(engine.step(settled, settings, 10.001), (0, 0))
        self.assertEqual(
            engine.step(
                TargetSnapshot(2, 100.0, "head", 166.0, 160.0),
                settings,
                10.002,
            ),
            (6, 0),
        )

    def test_distinct_reverse_target_uses_nominal_dt_after_emitted_settlement(self):
        engine = AimMovementEngine(nominal_hz=60)
        settings = AimSettings(
            aim_strength=1.0,
            smoothing=0.0,
            max_step=127,
            response_curve=self.LINEAR_CURVE,
        )

        self.assertEqual(
            engine.step(
                TargetSnapshot(1, 100.0, "head", 166.0, 160.0), settings, 10.0
            ),
            (6, 0),
        )
        self.assertEqual(
            engine.step(
                TargetSnapshot(2, 100.0, "head", 154.0, 160.0),
                settings,
                10.001,
            ),
            (-6, 0),
        )

    def test_public_reset_clears_settled_sequence_tombstone(self):
        engine = AimMovementEngine(nominal_hz=60)
        settings = AimSettings(
            aim_strength=1.0,
            smoothing=0.0,
            max_step=127,
            response_curve=self.LINEAR_CURVE,
        )
        target = TargetSnapshot(1, 100.0, "head", 166.0, 160.0)
        self.assertEqual(engine.step(target, settings, 10.0), (6, 0))

        engine.reset()

        self.assertEqual(engine.step(target, settings, 10.001), (6, 0))

    def test_none_after_settlement_clears_sequence_tombstone(self):
        engine = AimMovementEngine(nominal_hz=60)
        settings = AimSettings(
            aim_strength=1.0,
            smoothing=0.0,
            max_step=127,
            response_curve=self.LINEAR_CURVE,
        )
        target = TargetSnapshot(1, 100.0, "head", 166.0, 160.0)
        self.assertEqual(engine.step(target, settings, 10.0), (6, 0))

        self.assertEqual(engine.step(None, settings, 10.001), (0, 0))

        self.assertEqual(engine.step(target, settings, 10.002), (6, 0))

    def test_none_target_resets_all_motion_state(self):
        engine = AimMovementEngine(nominal_hz=60)
        settings = AimSettings(
            aim_strength=1.0,
            smoothing=0.95,
            max_step=127,
            response_curve=self.LINEAR_CURVE,
        )
        target = TargetSnapshot(1, 100.0, "head", 176.0, 160.0)
        engine.step(target, settings, 10.0)

        self.assertEqual(engine.step(None, settings, 10.1), (0, 0))
        self.assertEqual(
            engine.step(TargetSnapshot(2, 100.0, "head", 176.0, 160.0), settings, 10.2),
            (1, 0),
        )

    def test_backward_and_large_clock_jumps_are_clamped(self):
        engine = AimMovementEngine(nominal_hz=100)
        settings = AimSettings(
            aim_strength=1.0,
            smoothing=0.0,
            max_step=5,
            response_curve=self.LINEAR_CURVE,
        )
        target = TargetSnapshot(1, 100.0, "head", 320.0, 160.0)

        self.assertEqual(engine.step(target, settings, 10.0), (2, 0))
        self.assertEqual(engine.step(target, settings, 9.0), (0, 0))
        self.assertEqual(engine.step(target, settings, 11.0), (5, 0))

    def test_each_axis_cannot_overshoot_its_remaining_error(self):
        engine = AimMovementEngine(nominal_hz=10)
        settings = AimSettings(aim_strength=2.0, smoothing=0.0, max_step=127)
        engine.step(
            TargetSnapshot(1, 100.0, "head", 320.0, 320.0), settings, 10.0
        )

        report = engine.step(
            TargetSnapshot(2, 100.0, "head", 160.5, 320.0), settings, 10.1
        )

        self.assertEqual(report, (0, 127))

    def test_response_curve_controls_servo_speed(self):
        target = TargetSnapshot(1, 100.0, "head", 170.0, 160.0)
        common = dict(aim_strength=0.05, smoothing=0.0, max_step=127)

        stopped = AimMovementEngine(nominal_hz=10).step(
            target, AimSettings(response_curve=(0.0,) * 5, **common), 10.0
        )
        linear = AimMovementEngine(nominal_hz=10).step(
            target, AimSettings(response_curve=self.LINEAR_CURVE, **common), 10.0
        )

        self.assertEqual(stopped, (0, 0))
        self.assertEqual(linear, (3, 0))


if __name__ == "__main__":
    unittest.main()
