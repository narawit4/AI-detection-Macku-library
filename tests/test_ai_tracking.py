import math
import unittest
from dataclasses import FrozenInstanceError

from ai_targeting import AimSettings, Detection, TargetSnapshot
from ai_tracking import TrackerState, observe_detections


def head_box(center_x, center_y=100, size=10, confidence=0.9):
    half = size / 2
    return Detection(
        center_x - half,
        center_y - half,
        center_x + half,
        center_y + half,
        confidence,
        7,
    )


def head_rectangle(center_x, center_y, width, height, confidence=0.9):
    return Detection(
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
        confidence,
        7,
    )


def player_box(center_x, aim_y=100, width=20, height=100, confidence=0.9):
    half_width = width / 2
    top = aim_y - height * 0.20
    return Detection(
        center_x - half_width,
        top,
        center_x + half_width,
        top + height,
        confidence,
        0,
    )


def observe(state, detections, sequence, captured_at, settings=None):
    return observe_detections(
        state,
        detections,
        settings or AimSettings(),
        sequence=sequence,
        captured_at=captured_at,
    )


class ConservativeTrackingTests(unittest.TestCase):
    def test_crossing_does_not_follow_the_nearest_competitor(self):
        acquired = observe(
            TrackerState(), (head_box(130),), 1, 1 / 60
        )
        self.assertAlmostEqual(acquired.analysis.target.aim_x, 130)
        state = acquired.state
        observations = []
        for sequence, (person_a, person_b) in enumerate(
            ((140, 220), (150, 190), (160, 160), (175, 145), (190, 130)), 2
        ):
            result = observe_detections(
                state,
                (head_box(person_b), head_box(person_a)),
                AimSettings(),
                sequence=sequence,
                captured_at=sequence / 60,
            )
            state = result.state
            observations.append(result)
        self.assertAlmostEqual(observations[0].analysis.target.aim_x, 140)
        self.assertAlmostEqual(observations[1].analysis.target.aim_x, 150)
        self.assertIsNone(observations[2].analysis.target)
        self.assertEqual(observations[2].analysis.frame.selected_index, None)
        self.assertIsNone(observations[3].analysis.target)
        self.assertAlmostEqual(observations[-1].analysis.target.aim_x, 190)
        self.assertEqual(observations[-1].analysis.frame.selected_index, 1)

    def test_ambiguous_initial_frames_never_create_confirmed_geometry_or_velocity(self):
        state = TrackerState()
        for sequence in range(1, 4):
            result = observe(
                state,
                (head_box(180), head_box(140)),
                sequence,
                sequence / 60,
            )
            state = result.state
            self.assertIsNone(result.analysis.target)
            self.assertIsNone(result.analysis.frame.selected_index)
            self.assertIsNone(state.confirmed_detection)
            self.assertIsNone(state.confirmed_target)
            self.assertIsNone(state.preceding_target)
            self.assertIsNone(state.last_clear_at)
        self.assertEqual(state.pending_count, 3)

    def test_ambiguous_frame_keeps_every_current_accepted_box(self):
        state = observe(
            TrackerState(), (head_box(100),), 1, 0.0
        ).state
        low = head_box(80, confidence=0.2)
        left = head_box(99)
        right = head_box(101)

        result = observe(state, (low, left, right), 2, 0.010)

        self.assertIsNone(result.analysis.target)
        self.assertIsNone(result.analysis.frame.selected_index)
        self.assertEqual(result.analysis.frame.detections, (left, right))
        self.assertEqual(result.state.confirmed_target.aim_x, 100)

    def test_original_track_requires_two_clear_frames_after_ambiguity(self):
        state = observe(TrackerState(), (head_box(100),), 1, 0.0).state
        state = observe(state, (head_box(99), head_box(101)), 2, 0.010).state

        first = observe(state, (head_box(102),), 3, 0.020)
        second = observe(first.state, (head_box(103),), 4, 0.030)

        self.assertIsNone(first.analysis.target)
        self.assertIsNone(first.analysis.frame.selected_index)
        self.assertAlmostEqual(second.analysis.target.aim_x, 103)
        self.assertEqual(second.analysis.frame.selected_index, 0)

    def test_changed_recovery_candidate_restarts_two_frame_confirmation(self):
        state = observe(TrackerState(), (head_box(100),), 1, 0.0).state
        state = observe(state, (head_box(99), head_box(101)), 2, 0.010).state

        first = observe(state, (head_box(102),), 3, 0.020)
        changed = observe(first.state, (head_box(130),), 4, 0.030)
        confirmed = observe(changed.state, (head_box(131),), 5, 0.040)

        self.assertIsNone(first.analysis.target)
        self.assertIsNone(changed.analysis.target)
        self.assertEqual(changed.state.recovery_count, 1)
        self.assertAlmostEqual(confirmed.analysis.target.aim_x, 131)

    def test_recovery_started_before_expiry_can_finish_across_boundary(self):
        state = observe(TrackerState(), (head_box(100),), 1, 0.0).state
        state = observe(
            state, (head_box(99), head_box(101)), 2, 0.100
        ).state

        first = observe(state, (head_box(102),), 3, 0.149)
        second = observe(first.state, (head_box(103),), 4, 0.151)

        self.assertIsNone(first.analysis.target)
        self.assertEqual(first.state.recovery_count, 1)
        self.assertAlmostEqual(first.state.last_clear_at, 0.149)
        self.assertAlmostEqual(second.analysis.target.aim_x, 103)
        self.assertEqual(second.analysis.frame.selected_index, 0)

    def test_replacement_waits_for_three_stable_observations(self):
        state = TrackerState()
        first = observe_detections(
            state,
            (head_box(100),),
            AimSettings(),
            sequence=1,
            captured_at=0.0,
        )
        state = first.state
        results = []
        for sequence, captured_at in ((2, 0.150), (3, 0.167), (4, 0.184)):
            observed = observe_detections(
                state,
                (head_box(220),),
                AimSettings(),
                sequence=sequence,
                captured_at=captured_at,
            )
            state = observed.state
            results.append(observed.analysis.target)
        self.assertEqual(results[:2], [None, None])
        self.assertAlmostEqual(results[2].aim_x, 220)

    def test_exact_initial_score_margin_is_ambiguous(self):
        center_to_corner = math.hypot(160.0, 160.0)
        runner_x = 160.0 + 0.15 * center_to_corner

        result = observe(
            TrackerState(),
            (head_box(160, 160), head_box(runner_x, 160)),
            1,
            0.0,
        )

        self.assertIsNone(result.analysis.target)
        self.assertIsNone(result.analysis.frame.selected_index)

    def test_initial_score_gap_just_above_margin_is_clear(self):
        center_to_corner = math.hypot(160.0, 160.0)
        runner_x = 160.0 + 0.150001 * center_to_corner

        result = observe(
            TrackerState(),
            (head_box(160, 160), head_box(runner_x, 160)),
            1,
            0.0,
        )

        self.assertAlmostEqual(result.analysis.target.aim_x, 160)
        self.assertEqual(result.analysis.frame.selected_index, 0)

    def test_initial_ties_choose_deterministic_pending_candidate_across_input_order(self):
        first = observe(
            TrackerState(), (head_box(180), head_box(140)), 1, 0.0
        )
        reordered = observe(
            TrackerState(), (head_box(140), head_box(180)), 1, 0.0
        )

        self.assertAlmostEqual(first.state.pending_candidate.aim_x, 140)
        self.assertEqual(first.state.pending_count, 1)
        self.assertAlmostEqual(reordered.state.pending_candidate.aim_x, 140)
        self.assertEqual(reordered.state.pending_count, 1)
        self.assertIsNone(first.analysis.frame.selected_index)
        self.assertIsNone(reordered.analysis.frame.selected_index)

    def test_ambiguous_initial_choice_still_requires_three_total_observations(self):
        first = observe(
            TrackerState(), (head_box(180), head_box(140)), 1, 0.0
        )
        second = observe(first.state, (head_box(140),), 2, 0.010)
        third = observe(second.state, (head_box(140),), 3, 0.020)

        self.assertIsNone(second.analysis.target)
        self.assertIsNone(second.analysis.frame.selected_index)
        self.assertEqual(second.state.pending_count, 2)
        self.assertAlmostEqual(third.analysis.target.aim_x, 140)
        self.assertEqual(third.analysis.frame.selected_index, 0)

    def test_exact_tracked_composite_score_margin_is_ambiguous(self):
        confirmed_detection = head_rectangle(100, 100, 4, 1.75)
        confirmed_target = TargetSnapshot(1, 0.0, "head", 100, 100)
        state = TrackerState(
            confirmed_detection=confirmed_detection,
            confirmed_target=confirmed_target,
            last_clear_at=0.0,
        )
        exact_match = confirmed_detection
        exact_margin = head_rectangle(100, 100, 7, 1)

        result = observe(state, (exact_match, exact_margin), 2, 0.010)

        self.assertIsNone(result.analysis.target)
        self.assertIsNone(result.analysis.frame.selected_index)
        self.assertEqual(result.analysis.frame.detections, (exact_match, exact_margin))
        self.assertEqual(result.state.confirmed_detection, confirmed_detection)

    def test_plausibility_radius_is_inclusive(self):
        initial = observe(TrackerState(), (head_box(100),), 1, 0.0)

        boundary = observe(initial.state, (head_box(148),), 2, 0.010)
        outside = observe(
            initial.state,
            (head_box(148.000001),),
            2,
            0.010,
        )

        self.assertAlmostEqual(boundary.analysis.target.aim_x, 148)
        self.assertIsNone(outside.analysis.target)
        self.assertAlmostEqual(outside.state.confirmed_target.aim_x, 100)

    def test_large_box_plausibility_uses_one_and_a_half_times_diagonal(self):
        confirmed_detection = head_rectangle(100, 100, 40, 30)
        confirmed_target = TargetSnapshot(1, 0.0, "head", 100, 100)
        state = TrackerState(
            confirmed_detection=confirmed_detection,
            confirmed_target=confirmed_target,
            last_clear_at=0.0,
        )

        boundary = observe(
            state,
            (head_rectangle(175, 100, 40, 30),),
            2,
            0.010,
        )
        outside = observe(
            state,
            (head_rectangle(175.000001, 100, 40, 30),),
            2,
            0.010,
        )

        self.assertAlmostEqual(boundary.analysis.target.aim_x, 175)
        self.assertIsNone(outside.analysis.target)

    def test_area_ratio_boundaries_are_inclusive(self):
        initial = observe(TrackerState(), (head_box(100, size=10),), 1, 0.0)

        for ratio in (0.4, 2.5):
            with self.subTest(ratio=ratio):
                result = observe(
                    initial.state,
                    (head_rectangle(100, 100, 10 * ratio, 10),),
                    2,
                    0.010,
                )
                self.assertIsNotNone(result.analysis.target)

    def test_area_ratios_outside_boundaries_are_implausible(self):
        initial = observe(TrackerState(), (head_box(100, size=10),), 1, 0.0)

        for ratio in (0.399999, 2.500001):
            with self.subTest(ratio=ratio):
                result = observe(
                    initial.state,
                    (head_rectangle(100, 100, 10 * ratio, 10),),
                    2,
                    0.010,
                )
                self.assertIsNone(result.analysis.target)

    def test_velocity_prediction_is_capped_at_800_pixels_per_second(self):
        state = observe(TrackerState(), (head_box(100),), 1, 0.0).state
        state = observe(state, (head_box(140),), 2, 0.010).state

        result = observe(state, (head_box(100),), 3, 0.020)

        self.assertAlmostEqual(result.analysis.target.aim_x, 100)

    def test_prediction_horizon_is_capped_at_100_milliseconds(self):
        state = observe(TrackerState(), (head_box(100),), 1, 0.0).state
        state = observe(state, (head_box(140),), 2, 0.050).state

        result = observe(state, (head_box(172),), 3, 0.160)

        self.assertAlmostEqual(result.analysis.target.aim_x, 172)

    def test_predictive_hold_expires_at_exactly_150_milliseconds(self):
        initial = observe(TrackerState(), (head_box(100),), 1, 0.0)

        before = observe(initial.state, (head_box(220),), 2, 0.149999)
        expired = observe(initial.state, (head_box(220),), 2, 0.150)

        self.assertEqual(before.state.pending_count, 0)
        self.assertEqual(before.state.confirmed_target, initial.state.confirmed_target)
        self.assertEqual(expired.state.pending_count, 1)
        self.assertIsNone(expired.analysis.target)

    def test_pending_displacement_boundary_is_inclusive(self):
        state = observe(TrackerState(), (head_box(100),), 1, 0.0).state
        first = observe(state, (head_box(220),), 2, 0.150)
        second = observe(first.state, (head_box(238),), 3, 0.167)
        third = observe(second.state, (head_box(256),), 4, 0.184)

        self.assertEqual(first.state.pending_count, 1)
        self.assertEqual(second.state.pending_count, 2)
        self.assertAlmostEqual(third.analysis.target.aim_x, 256)

    def test_pending_displacement_beyond_boundary_restarts_confirmation(self):
        state = observe(TrackerState(), (head_box(100),), 1, 0.0).state
        first = observe(state, (head_box(220),), 2, 0.150)
        second = observe(
            first.state,
            (head_box(238.000001),),
            3,
            0.167,
        )

        self.assertEqual(second.state.pending_count, 1)

    def test_initial_acquisition_prefers_head_over_centered_player(self):
        player = player_box(160, 160)
        head = head_box(300, 300)

        result = observe(TrackerState(), (player, head), 1, 0.0)

        self.assertEqual(result.analysis.target.target_class, "head")
        self.assertEqual(result.analysis.frame.selected_index, 1)

    def test_replacement_selection_prefers_head_over_centered_player(self):
        state = observe(TrackerState(), (head_box(100),), 1, 0.0).state
        detections = (player_box(160, 160), head_box(280, 280))

        first = observe(state, detections, 2, 0.150)
        second = observe(first.state, detections, 3, 0.167)
        third = observe(second.state, detections, 4, 0.184)

        self.assertIsNone(first.analysis.target)
        self.assertIsNone(second.analysis.target)
        self.assertEqual(third.analysis.target.target_class, "head")
        self.assertEqual(third.analysis.frame.selected_index, 1)

    def test_malformed_and_unaccepted_detections_are_not_published(self):
        malformed = (
            Detection(1, 1, 1, 10, 0.9, 7),
            Detection(10, 1, 1, 10, 0.9, 7),
            Detection(10, 10, 1, 1, 0.9, 7),
            Detection(math.nan, 1, 10, 10, 0.9, 7),
            Detection(1, 1, 10, 10, math.inf, 7),
            Detection(1, 1, 10, 10, 0.9, 99),
            head_box(120, confidence=0.34),
        )
        valid = head_box(160, 160)

        result = observe(TrackerState(), (*malformed, valid), 1, 0.0)

        self.assertEqual(result.analysis.frame.detections, (valid,))
        self.assertEqual(result.analysis.frame.selected_index, 0)
        self.assertAlmostEqual(result.analysis.target.aim_x, 160)

    def test_current_selected_index_uses_the_accepted_tuple(self):
        low = head_box(10, confidence=0.2)
        player = player_box(160, 160)
        head = head_box(300, 300)

        result = observe(TrackerState(), (low, player, head), 1, 0.0)

        self.assertEqual(result.analysis.frame.detections, (player, head))
        self.assertEqual(result.analysis.frame.selected_index, 1)

    def test_missing_frame_never_publishes_a_held_coordinate(self):
        initial = observe(TrackerState(), (head_box(100),), 1, 0.0)

        missing = observe(initial.state, (), 2, 0.010)

        self.assertIsNone(missing.analysis.target)
        self.assertIsNone(missing.analysis.frame.selected_index)
        self.assertEqual(missing.state.confirmed_target, initial.state.confirmed_target)

    def test_tracker_records_are_frozen_and_observations_are_independent(self):
        empty = TrackerState()
        first = observe(empty, (head_box(100),), 1, 0.0)
        second = observe(empty, (head_box(220),), 1, 0.0)

        self.assertIsNone(empty.confirmed_target)
        self.assertAlmostEqual(first.state.confirmed_target.aim_x, 100)
        self.assertAlmostEqual(second.state.confirmed_target.aim_x, 220)
        self.assertIsNot(first.state, second.state)
        with self.assertRaises(FrozenInstanceError):
            first.state.pending_count = 99
        with self.assertRaises(FrozenInstanceError):
            first.analysis.target = None


if __name__ == "__main__":
    unittest.main()
