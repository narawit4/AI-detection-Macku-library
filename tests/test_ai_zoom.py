import math
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from jitter_app.ai.targeting import (
    AimSettings,
    Detection,
    DetectionAnalysis,
    DetectionFrameSnapshot,
    TargetSnapshot,
)
from jitter_app.ai.zoom import (
    RECOIL_COOLDOWN_SECONDS,
    ZoomStabilityState,
    ZoomTransform,
    build_zoom_input,
    compose_zoom_refinement,
    map_detection,
    map_target,
    limit_zoom_factor,
    movement_is_confirmed,
    observe_zoom_stability,
    record_zoom_refinement_miss,
    select_zoom_factor,
)


class ZoomStabilityTests(unittest.TestCase):
    def target(self, sequence=1, kind="head", x=100.0, y=100.0):
        return TargetSnapshot(sequence, 10.0, kind, x, y)

    def test_initial_state_is_empty_unconfirmed_and_immutable(self):
        state = ZoomStabilityState()
        self.assertIsNone(state.previous_base_target)
        self.assertEqual(state.stable_count, 0)
        self.assertEqual(state.cooldown_until, 0.0)
        self.assertFalse(movement_is_confirmed(state))
        with self.assertRaises(FrozenInstanceError):
            state.stable_count = 1

    def test_first_acquisition_starts_count_and_cooldown(self):
        target = self.target()
        state = observe_zoom_stability(
            ZoomStabilityState(), target, 10.0
        )
        self.assertEqual(state.previous_base_target, target)
        self.assertEqual(state.stable_count, 1)
        self.assertAlmostEqual(
            state.cooldown_until, 10.0 + RECOIL_COOLDOWN_SECONDS
        )
        self.assertFalse(movement_is_confirmed(state))

    def test_exact_18_pixels_confirms_but_18_point_01_restarts(self):
        first = observe_zoom_stability(
            ZoomStabilityState(), self.target(x=100.0), 10.0
        )
        exact = observe_zoom_stability(
            first, self.target(sequence=2, x=118.0), 10.01
        )
        outside = observe_zoom_stability(
            first, self.target(sequence=2, x=118.01), 10.01
        )
        self.assertEqual(exact.stable_count, 2)
        self.assertTrue(movement_is_confirmed(exact))
        self.assertEqual(outside.stable_count, 1)
        self.assertFalse(movement_is_confirmed(outside))
        self.assertAlmostEqual(outside.cooldown_until, 10.11)

    def test_zoom_stability_scales_native_displacement_to_logical_320(self):
        first = TargetSnapshot(1, 10.0, "head", 960, 540, 1920, 1080)
        exact = TargetSnapshot(2, 10.01, "head", 1068, 540, 1920, 1080)
        outside = TargetSnapshot(
            3, 10.02, "head", 1068.1, 540, 1920, 1080
        )
        state = observe_zoom_stability(ZoomStabilityState(), first, 10.0)
        self.assertEqual(
            observe_zoom_stability(state, exact, 10.01).stable_count, 2
        )
        self.assertEqual(
            observe_zoom_stability(state, outside, 10.02).stable_count, 1
        )

    def test_zoom_stability_resets_when_frame_geometry_changes(self):
        first = TargetSnapshot(1, 10.0, "head", 960, 540, 320, 320)
        changed = TargetSnapshot(2, 10.01, "head", 960, 540, 1920, 1080)
        state = observe_zoom_stability(ZoomStabilityState(), first, 10.0)

        result = observe_zoom_stability(state, changed, 10.01)

        self.assertEqual(result.previous_base_target, changed)
        self.assertEqual(result.stable_count, 1)
        self.assertAlmostEqual(result.cooldown_until, 10.11)

    def test_class_change_is_unstable_inside_distance_boundary(self):
        first = observe_zoom_stability(
            ZoomStabilityState(), self.target(kind="player"), 10.0
        )
        changed = observe_zoom_stability(
            first,
            self.target(sequence=2, kind="head", x=101.0),
            10.02,
        )
        self.assertEqual(changed.stable_count, 1)
        self.assertEqual(changed.previous_base_target.target_class, "head")
        self.assertAlmostEqual(changed.cooldown_until, 10.12)

    def test_missing_target_clears_previous_and_confirmation(self):
        first = observe_zoom_stability(
            ZoomStabilityState(), self.target(), 10.0
        )
        confirmed = observe_zoom_stability(
            first, self.target(sequence=2), 10.01
        )
        missing = observe_zoom_stability(confirmed, None, 10.02)
        self.assertIsNone(missing.previous_base_target)
        self.assertEqual(missing.stable_count, 0)
        self.assertFalse(movement_is_confirmed(missing))
        self.assertAlmostEqual(missing.cooldown_until, 10.12)

    def test_two_x_requires_confirmation_and_exact_cooldown_boundary(self):
        first = observe_zoom_stability(
            ZoomStabilityState(), self.target(), 10.0
        )
        confirmed = observe_zoom_stability(
            first, self.target(sequence=2), 10.02
        )
        self.assertEqual(limit_zoom_factor(1.0, first, 10.0), 1.0)
        self.assertEqual(limit_zoom_factor(1.5, first, 10.0), 1.5)
        self.assertEqual(limit_zoom_factor(2.0, first, 10.2), 1.5)
        self.assertEqual(limit_zoom_factor(2.0, confirmed, 10.099), 1.5)
        self.assertEqual(limit_zoom_factor(2.0, confirmed, 10.1), 2.0)

    def test_refinement_miss_resets_count_keeps_target_and_extends_only(self):
        target = self.target()
        state = ZoomStabilityState(target, 2, 20.0)
        retained = record_zoom_refinement_miss(state, 10.0)
        extended = record_zoom_refinement_miss(retained, 20.0)
        self.assertEqual(retained.previous_base_target, target)
        self.assertEqual(retained.stable_count, 0)
        self.assertEqual(retained.cooldown_until, 20.0)
        self.assertEqual(extended.previous_base_target, target)
        self.assertEqual(extended.stable_count, 0)
        self.assertAlmostEqual(extended.cooldown_until, 20.1)


class ZoomFactorTests(unittest.TestCase):
    def target(self, kind="head", x=160.0, y=160.0):
        return TargetSnapshot(1, 10.0, kind, x, y)

    def test_head_height_boundaries_select_exact_factors(self):
        cases = (
            (18.0, 2.0),
            (18.01, 1.5),
            (32.0, 1.5),
            (32.01, 1.0),
        )
        for height, expected in cases:
            with self.subTest(height=height):
                detection = Detection(150, 100, 170, 100 + height, 0.9, 7)
                self.assertEqual(
                    select_zoom_factor(detection, self.target()), expected
                )

    def test_player_height_boundaries_select_exact_factors(self):
        cases = (
            (64.0, 2.0),
            (64.01, 1.5),
            (112.0, 1.5),
            (112.01, 1.0),
        )
        for height, expected in cases:
            with self.subTest(height=height):
                detection = Detection(140, 80, 180, 80 + height, 0.9, 0)
                self.assertEqual(
                    select_zoom_factor(
                        detection, self.target("player")
                    ),
                    expected,
                )

    def test_missing_unsupported_or_outside_center_target_stays_one_x(self):
        head = Detection(150, 100, 170, 110, 0.9, 7)
        unsupported = Detection(150, 100, 170, 110, 0.9, 4)
        self.assertEqual(select_zoom_factor(head, None), 1.0)
        self.assertEqual(
            select_zoom_factor(unsupported, self.target()), 1.0
        )
        self.assertEqual(
            select_zoom_factor(head, self.target(x=256.01)), 1.0
        )
        self.assertEqual(
            select_zoom_factor(head, self.target(x=256.0)), 2.0
        )

    def test_zoom_factor_uses_resolution_independent_logical_policy_space(self):
        legacy_target = TargetSnapshot(1, 10.0, "head", 160, 160)
        native_target = TargetSnapshot(
            2, 10.0, "head", 960, 540, 1920, 1080
        )
        legacy_box = Detection(150, 150, 170, 168, 0.9, 7)
        native_box = Detection(900, 480, 1020, 588, 0.9, 7)
        self.assertEqual(
            select_zoom_factor(legacy_box, legacy_target), 2.0
        )
        self.assertEqual(
            select_zoom_factor(native_box, native_target), 2.0
        )


class ZoomGeometryTests(unittest.TestCase):
    def test_two_x_crop_clamps_at_top_left_and_maps_coordinates_back(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        target = TargetSnapshot(4, 20.0, "head", 20.0, 30.0)
        zoomed, transform = build_zoom_input(frame, target, 2.0)
        self.assertEqual(
            transform, ZoomTransform(0, 0, 160, 160, 320, 320, 2.0)
        )
        self.assertEqual(zoomed.shape, (160, 160, 3))
        self.assertEqual(zoomed.dtype, np.uint8)
        self.assertFalse(np.shares_memory(zoomed, frame))
        mapped = map_detection(
            Detection(20, 40, 100, 140, 0.8, 7), transform
        )
        self.assertEqual(
            mapped, Detection(20, 40, 100, 140, 0.8, 7)
        )
        self.assertEqual(
            map_target(
                TargetSnapshot(4, 20.0, "head", 80, 120, 160, 160),
                transform,
            ),
            TargetSnapshot(4, 20.0, "head", 80, 120, 320, 320),
        )

    def test_full_hd_two_x_crop_preserves_aspect_and_translates_back(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        target = TargetSnapshot(
            4, 20.0, "head", 960.0, 540.0, 1920, 1080
        )

        crop, transform = build_zoom_input(frame, target, 2.0)

        self.assertEqual(crop.shape, (540, 960, 3))
        self.assertTrue(crop.flags.c_contiguous)
        self.assertTrue(crop.flags.owndata)
        self.assertEqual(
            transform,
            ZoomTransform(480, 270, 960, 540, 1920, 1080, 2.0),
        )
        self.assertEqual(
            map_detection(
                Detection(10, 20, 110, 220, 0.9, 7), transform
            ),
            Detection(490, 290, 590, 490, 0.9, 7),
        )

    def test_native_crop_sizes_use_half_up_rounding_at_each_zoom_factor(self):
        cases = (
            ((1080, 1920), 1.5, (720, 1280)),
            ((1079, 1919), 2.0, (540, 960)),
        )
        for (height, width), factor, expected_shape in cases:
            with self.subTest(size=(width, height), factor=factor):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                target = TargetSnapshot(
                    4,
                    20.0,
                    "head",
                    width / 2.0,
                    height / 2.0,
                    width,
                    height,
                )
                crop, transform = build_zoom_input(frame, target, factor)
                self.assertEqual(crop.shape[:2], expected_shape)
                self.assertEqual(
                    (transform.crop_width, transform.crop_height),
                    (expected_shape[1], expected_shape[0]),
                )

    def test_native_crop_rejects_target_from_different_frame_geometry(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        stale_target = TargetSnapshot(4, 20.0, "head", 160, 160)
        with self.assertRaisesRegex(ValueError, "match zoom source"):
            build_zoom_input(frame, stale_target, 2.0)

    def test_native_zoom_crop_clamps_each_source_edge(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cases = (
            ((20.0, 20.0), (0, 0)),
            ((1900.0, 20.0), (960, 0)),
            ((20.0, 1060.0), (0, 540)),
            ((1900.0, 1060.0), (960, 540)),
        )
        for (aim_x, aim_y), (left, top) in cases:
            with self.subTest(aim=(aim_x, aim_y)):
                target = TargetSnapshot(
                    5, 21.0, "head", aim_x, aim_y, 1920, 1080
                )
                _crop, transform = build_zoom_input(frame, target, 2.0)
                self.assertEqual((transform.left, transform.top), (left, top))

    def test_one_half_x_crop_clamps_at_bottom_right(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        target = TargetSnapshot(5, 21.0, "player", 310.0, 300.0)
        _zoomed, transform = build_zoom_input(frame, target, 1.5)
        self.assertEqual(
            transform, ZoomTransform(107, 107, 213, 213, 320, 320, 1.5)
        )

    def test_crop_sizes_centering_and_all_four_edge_clamps(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        cases = (
            (
                2.0,
                160.0,
                160.0,
                ZoomTransform(80, 80, 160, 160, 320, 320, 2.0),
            ),
            (
                1.5,
                160.0,
                160.0,
                ZoomTransform(54, 54, 213, 213, 320, 320, 1.5),
            ),
            (
                2.0,
                310.0,
                20.0,
                ZoomTransform(160, 0, 160, 160, 320, 320, 2.0),
            ),
            (
                2.0,
                20.0,
                300.0,
                ZoomTransform(0, 160, 160, 160, 320, 320, 2.0),
            ),
            (
                2.0,
                310.0,
                300.0,
                ZoomTransform(160, 160, 160, 160, 320, 320, 2.0),
            ),
        )
        for factor, aim_x, aim_y, expected in cases:
            with self.subTest(factor=factor, aim_x=aim_x, aim_y=aim_y):
                target = TargetSnapshot(6, 22.0, "head", aim_x, aim_y)
                _zoomed, transform = build_zoom_input(frame, target, factor)
                self.assertEqual(transform, expected)

    def test_mapping_clamps_nonempty_box_to_source_bounds(self):
        transform = ZoomTransform(280, 280, 40, 40, 320, 320, 8.0)
        self.assertEqual(
            map_detection(
                Detection(-100, -100, 400, 400, 0.8, 0), transform
            ),
            Detection(180.0, 180.0, 320.0, 320.0, 0.8, 0),
        )

    def test_mapping_discards_empty_box_after_clamping(self):
        transform = ZoomTransform(0, 0, 40, 40, 320, 320, 8.0)
        self.assertIsNone(
            map_detection(
                Detection(-100, -100, -1, -1, 0.8, 0), transform
            )
        )


class ZoomCompositionTests(unittest.TestCase):
    def base_player(self):
        return DetectionAnalysis(
            TargetSnapshot(7, 30.0, "player", 160.0, 120.0),
            DetectionFrameSnapshot(
                7,
                30.0,
                (
                    Detection(20, 40, 60, 140, 0.7, 0),
                    Detection(140, 80, 180, 280, 0.8, 0),
                    Detection(250, 40, 300, 160, 0.75, 0),
                ),
                1,
            ),
        )

    def test_player_seed_refines_to_mapped_head_and_preserves_other_boxes(self):
        result = compose_zoom_refinement(
            self.base_player(),
            (Detection(70, 35, 90, 55, 0.92, 7),),
            ZoomTransform(80, 40, 160, 160, 320, 320, 2.0),
            AimSettings(confidence=0.35),
        )
        self.assertIsNotNone(result)
        self.assertEqual(
            result.target,
            TargetSnapshot(7, 30.0, "head", 160.0, 85.0),
        )
        self.assertEqual(result.frame.selected_index, 1)
        self.assertEqual(result.frame.detections[0], self.base_player().frame.detections[0])
        self.assertEqual(
            result.frame.detections[1],
            Detection(150, 75, 170, 95, 0.92, 7),
        )
        self.assertEqual(result.frame.detections[2], self.base_player().frame.detections[2])

    def test_multiple_compatible_refinements_fail_back_to_base(self):
        result = compose_zoom_refinement(
            self.base_player(),
            (
                Detection(70, 35, 90, 55, .92, 7),
                Detection(74, 39, 94, 59, .93, 7),
            ),
            ZoomTransform(80, 40, 160, 160, 320, 320, 2.0),
            AimSettings(confidence=.35),
        )
        self.assertIsNone(result)

    def test_refinement_preserves_capture_viewport(self):
        original = self.base_player()
        base = DetectionAnalysis(
            original.target,
            DetectionFrameSnapshot(
                original.frame.sequence,
                original.frame.captured_at,
                original.frame.detections,
                original.frame.selected_index,
                original.frame.frame_width,
                original.frame.frame_height,
                1920,
                1080,
                800,
                380,
            ),
        )

        result = compose_zoom_refinement(
            base,
            (Detection(70, 35, 90, 55, 0.92, 7),),
            ZoomTransform(80, 40, 160, 160, 320, 320, 2.0),
            AimSettings(confidence=0.35),
        )

        self.assertIsNotNone(result)
        self.assertEqual(
            (
                result.frame.output_width,
                result.frame.output_height,
                result.frame.capture_left,
                result.frame.capture_top,
            ),
            (1920, 1080, 800, 380),
        )

    def test_refinement_stays_with_selected_base_target_in_crowded_crop(self):
        base = DetectionAnalysis(
            TargetSnapshot(8, 31.0, "head", 100.0, 100.0),
            DetectionFrameSnapshot(
                8,
                31.0,
                (Detection(70, 70, 130, 130, 0.9, 7),),
                0,
            ),
        )
        matching_base = Detection(35, 35, 45, 45, 0.92, 7)
        nearer_crosshair = Detection(85, 85, 95, 95, 0.95, 7)

        result = compose_zoom_refinement(
            base,
            (nearer_crosshair, matching_base),
            ZoomTransform(60, 60, 160, 160, 320, 320, 2.0),
            AimSettings(confidence=0.35),
        )

        self.assertIsNotNone(result)
        self.assertEqual((result.target.aim_x, result.target.aim_y), (100.0, 100.0))
        self.assertEqual(
            result.frame.detections[0],
            Detection(95.0, 95.0, 105.0, 105.0, 0.92, 7),
        )

    def test_body_target_area_rejects_head_only_refinement(self):
        result = compose_zoom_refinement(
            self.base_player(),
            (Detection(70, 35, 90, 55, 0.92, 7),),
            ZoomTransform(80, 40, 160, 160, 320, 320, 2.0),
            AimSettings(confidence=0.35, target_area="upper_body"),
        )

        self.assertIsNone(result)

    def test_head_seed_rejects_player_downgrade(self):
        base = DetectionAnalysis(
            TargetSnapshot(8, 31.0, "head", 160.0, 100.0),
            DetectionFrameSnapshot(
                8, 31.0, (Detection(150, 90, 170, 110, 0.9, 7),), 0
            ),
        )
        self.assertIsNone(
            compose_zoom_refinement(
                base,
                (Detection(50, 40, 110, 150, 0.95, 0),),
                ZoomTransform(80, 40, 160, 160, 320, 320, 2.0),
                AimSettings(confidence=0.35),
            )
        )

    def test_outside_expanded_seed_and_low_confidence_fall_back(self):
        for detection in (
            Detection(150, 5, 159, 15, 0.9, 7),
            Detection(70, 35, 90, 55, 0.1, 7),
        ):
            with self.subTest(detection=detection):
                self.assertIsNone(
                    compose_zoom_refinement(
                        self.base_player(),
                        (detection,),
                        ZoomTransform(
                            80, 40, 160, 160, 320, 320, 2.0
                        ),
                        AimSettings(confidence=0.35),
                    )
                )

    def test_association_includes_exact_12px_and_20_percent_margins(self):
        transform = ZoomTransform(80, 0, 160, 160, 320, 320, 2.0)
        exact_boundary = Detection(38, 30, 58, 50, 0.92, 7)
        result = compose_zoom_refinement(
            self.base_player(),
            (exact_boundary,),
            transform,
            AimSettings(confidence=0.35),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.target.aim_x, 128.0)
        self.assertEqual(result.target.aim_y, 40.0)

        for just_outside in (
            Detection(37.99, 30, 57.99, 50, 0.92, 7),
            Detection(38, 29.99, 58, 49.99, 0.92, 7),
        ):
            with self.subTest(just_outside=just_outside):
                self.assertIsNone(
                    compose_zoom_refinement(
                        self.base_player(),
                        (just_outside,),
                        transform,
                        AimSettings(confidence=0.35),
                    )
                )

    def test_native_association_boundary_and_composition_preserve_geometry(self):
        cases = (
            (1920, 1080, 480, 270),
            (1080, 1920, 270, 480),
        )
        for width, height, left, top in cases:
            with self.subTest(geometry=(width, height)):
                center_x = width / 2.0
                center_y = height / 2.0
                seed = Detection(
                    center_x - 60,
                    center_y - 60,
                    center_x + 60,
                    center_y + 60,
                    0.9,
                    7,
                )
                unrelated = Detection(20, 40, 60, 140, 0.7, 0)
                base = DetectionAnalysis(
                    TargetSnapshot(
                        9,
                        32.0,
                        "head",
                        center_x,
                        center_y,
                        width,
                        height,
                    ),
                    DetectionFrameSnapshot(
                        9,
                        32.0,
                        (unrelated, seed),
                        1,
                        width,
                        height,
                    ),
                )
                transform = ZoomTransform(
                    left,
                    top,
                    math.floor(width / 2.0 + 0.5),
                    math.floor(height / 2.0 + 0.5),
                    width,
                    height,
                    2.0,
                )
                accepted_aim_x = seed.x1 - 72.0
                accepted = Detection(
                    accepted_aim_x - left - 10,
                    center_y - top - 10,
                    accepted_aim_x - left + 10,
                    center_y - top + 10,
                    0.92,
                    7,
                )

                result = compose_zoom_refinement(
                    base,
                    (accepted,),
                    transform,
                    AimSettings(confidence=0.35),
                )

                self.assertIsNotNone(result)
                self.assertEqual(
                    (result.target.frame_width, result.target.frame_height),
                    (width, height),
                )
                self.assertEqual(
                    (result.frame.frame_width, result.frame.frame_height),
                    (width, height),
                )
                self.assertIs(result.frame.detections[0], unrelated)
                self.assertEqual(
                    result.frame.detections[1],
                    Detection(
                        accepted_aim_x - 10,
                        center_y - 10,
                        accepted_aim_x + 10,
                        center_y + 10,
                        0.92,
                        7,
                    ),
                )
                just_outside_aim_x = accepted_aim_x - 0.01
                just_outside = Detection(
                    just_outside_aim_x - left - 10,
                    center_y - top - 10,
                    just_outside_aim_x - left + 10,
                    center_y - top + 10,
                    0.92,
                    7,
                )
                self.assertIsNone(
                    compose_zoom_refinement(
                        base,
                        (just_outside,),
                        transform,
                        AimSettings(confidence=0.35),
                    )
                )


if __name__ == "__main__":
    unittest.main()
