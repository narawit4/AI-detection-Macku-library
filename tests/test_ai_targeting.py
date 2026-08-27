import math
import unittest
from dataclasses import FrozenInstanceError

from ai_targeting import (
    AIM_LIMITS,
    AimSettings,
    DEFAULT_RESPONSE_CURVE,
    Detection,
    DetectionFrameSnapshot,
    RESPONSE_CURVE_X,
    TargetLockState,
    TargetSnapshot,
    AimMovementEngine,
    aim_settings_from_mapping,
    aim_settings_to_mapping,
    analyze_detections,
    detection_aim_point,
    observe_target_lock,
    response_curve_value,
    select_target,
    target_lock_allows,
    validated_response_curve,
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


class TargetLockTests(unittest.TestCase):
    def target(self, sequence, x, *, kind="head", y=160.0):
        return TargetSnapshot(sequence, 10.0 + sequence / 100.0, kind, x, y)

    def test_initial_target_is_confirmed_and_state_is_immutable(self):
        target = self.target(1, 80.0)

        state = observe_target_lock(TargetLockState(), target)

        self.assertEqual(state.confirmed_target, target)
        self.assertIsNone(state.pending_target)
        self.assertEqual(state.pending_count, 0)
        self.assertTrue(target_lock_allows(state, target))
        with self.assertRaises(FrozenInstanceError):
            state.pending_count = 1

    def test_unassociated_target_requires_three_stable_observations(self):
        original = self.target(1, 80.0)
        state = TargetLockState(confirmed_target=original)
        first = self.target(2, 155.0)
        second = self.target(3, 156.0)
        third = self.target(4, 154.0)

        state = observe_target_lock(state, first)
        self.assertEqual(state.confirmed_target, original)
        self.assertEqual(state.pending_count, 1)
        self.assertFalse(target_lock_allows(state, first))

        state = observe_target_lock(state, second)
        self.assertEqual(state.confirmed_target, original)
        self.assertEqual(state.pending_count, 2)
        self.assertFalse(target_lock_allows(state, second))

        state = observe_target_lock(state, third)
        self.assertEqual(state.confirmed_target, third)
        self.assertIsNone(state.pending_target)
        self.assertEqual(state.pending_count, 0)
        self.assertTrue(target_lock_allows(state, third))

    def test_pending_candidate_change_restarts_confirmation(self):
        original = self.target(1, 80.0)
        state = observe_target_lock(
            TargetLockState(confirmed_target=original),
            self.target(2, 155.0),
        )

        changed = self.target(3, 130.0)
        state = observe_target_lock(state, changed)

        self.assertEqual(state.confirmed_target, original)
        self.assertEqual(state.pending_target, changed)
        self.assertEqual(state.pending_count, 1)
        self.assertFalse(target_lock_allows(state, changed))

    def test_original_target_return_cancels_pending_switch(self):
        original = self.target(1, 80.0)
        state = observe_target_lock(
            TargetLockState(confirmed_target=original),
            self.target(2, 155.0),
        )

        returned = self.target(3, 82.0)
        state = observe_target_lock(state, returned)

        self.assertEqual(state.confirmed_target, returned)
        self.assertIsNone(state.pending_target)
        self.assertEqual(state.pending_count, 0)
        self.assertTrue(target_lock_allows(state, returned))

    def test_missing_target_keeps_anchor_without_allowing_movement(self):
        original = self.target(1, 80.0)
        state = observe_target_lock(
            TargetLockState(confirmed_target=original),
            self.target(2, 155.0),
        )

        state = observe_target_lock(state, None)

        self.assertEqual(state.confirmed_target, original)
        self.assertIsNone(state.pending_target)
        self.assertEqual(state.pending_count, 0)
        self.assertFalse(target_lock_allows(state, None))


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
    LINEAR_CURVE = (0.0, 0.25, 0.5, 0.75, 1.0)

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
        for hz in (120, 288, 480):
            engine = AimMovementEngine(nominal_hz=hz)
            target = TargetSnapshot(1, 20.0, "head", 220.0, 160.0)
            reports = [
                engine.step(target, AimSettings(smoothing=0.0), 20 + index / hz)
                for index in range(int(hz * 0.1))
            ]
            totals.append(sum(x for x, _ in reports))
        self.assertLessEqual(max(totals) - min(totals), 2)

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
