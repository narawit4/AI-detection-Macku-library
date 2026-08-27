import hashlib
import unittest

import numpy as np

import ai_zoom
from ai_targeting import (
    AimSettings,
    Detection,
    DetectionAnalysis,
    DetectionFrameSnapshot,
    TargetSnapshot,
)
from ai_zoom import (
    ZoomTransform,
    build_zoom_input,
    compose_zoom_refinement,
    map_detection,
    map_target,
    resize_rgb_bilinear,
    select_zoom_factor,
)


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


class ZoomGeometryTests(unittest.TestCase):
    def test_resize_reuses_cached_coordinate_plan(self):
        plan_builder = getattr(ai_zoom, "_resize_plan", None)
        self.assertTrue(
            callable(plan_builder),
            "resize must expose its internal cached coordinate plan",
        )
        plan_builder.cache_clear()
        first = plan_builder(160, 160, 320)
        second = plan_builder(160, 160, 320)
        self.assertIs(first, second)

    def test_zoom_size_resizes_match_frozen_pixel_outputs(self):
        expected_hashes = {
            160: "73fff29a5890f0cdad009470c7ade901489fcc096309af53489d1812cf23e5a5",
            213: "d3876b8a91d71fcd2303a1f5e94c551761191e534b8861c623a908a1c3a4bd32",
        }
        random = np.random.default_rng(20260827)
        for size, expected_hash in expected_hashes.items():
            with self.subTest(size=size):
                source = random.integers(
                    0, 256, (size, size, 3), dtype=np.uint8
                )
                resized = resize_rgb_bilinear(source)
                self.assertEqual(
                    hashlib.sha256(resized.tobytes()).hexdigest(),
                    expected_hash,
                )

    def test_bilinear_resize_has_hand_derived_center_and_owned_rgb_output(self):
        source = np.array(
            [
                [[0, 0, 0], [100, 100, 100]],
                [[200, 200, 200], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )
        resized = resize_rgb_bilinear(source, output_size=3)
        self.assertEqual(resized.shape, (3, 3, 3))
        self.assertEqual(resized.dtype, np.uint8)
        self.assertEqual(resized[1, 1].tolist(), [139, 139, 139])
        self.assertFalse(np.shares_memory(resized, source))

    def test_two_x_crop_clamps_at_top_left_and_maps_coordinates_back(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        target = TargetSnapshot(4, 20.0, "head", 20.0, 30.0)
        zoomed, transform = build_zoom_input(frame, target, 2.0)
        self.assertEqual(transform, ZoomTransform(0, 0, 160, 2.0))
        self.assertEqual(zoomed.shape, (320, 320, 3))
        self.assertEqual(zoomed.dtype, np.uint8)
        self.assertFalse(np.shares_memory(zoomed, frame))
        mapped = map_detection(
            Detection(20, 40, 100, 140, 0.8, 7), transform
        )
        self.assertEqual(
            mapped, Detection(10, 20, 50, 70, 0.8, 7)
        )
        self.assertEqual(
            map_target(
                TargetSnapshot(4, 20.0, "head", 80, 120), transform
            ),
            TargetSnapshot(4, 20.0, "head", 40, 60),
        )

    def test_one_half_x_crop_clamps_at_bottom_right(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        target = TargetSnapshot(5, 21.0, "player", 310.0, 300.0)
        _zoomed, transform = build_zoom_input(frame, target, 1.5)
        self.assertEqual(transform, ZoomTransform(107, 107, 213, 1.5))

    def test_crop_sizes_centering_and_all_four_edge_clamps(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        cases = (
            (2.0, 160.0, 160.0, ZoomTransform(80, 80, 160, 2.0)),
            (1.5, 160.0, 160.0, ZoomTransform(54, 54, 213, 1.5)),
            (2.0, 310.0, 20.0, ZoomTransform(160, 0, 160, 2.0)),
            (2.0, 20.0, 300.0, ZoomTransform(0, 160, 160, 2.0)),
            (2.0, 310.0, 300.0, ZoomTransform(160, 160, 160, 2.0)),
        )
        for factor, aim_x, aim_y, expected in cases:
            with self.subTest(factor=factor, aim_x=aim_x, aim_y=aim_y):
                target = TargetSnapshot(6, 22.0, "head", aim_x, aim_y)
                _zoomed, transform = build_zoom_input(frame, target, factor)
                self.assertEqual(transform, expected)

    def test_mapping_clamps_nonempty_box_to_source_bounds(self):
        transform = ZoomTransform(280, 280, 40, 8.0)
        self.assertEqual(
            map_detection(
                Detection(-100, -100, 400, 400, 0.8, 0), transform
            ),
            Detection(267.5, 267.5, 320.0, 320.0, 0.8, 0),
        )

    def test_mapping_discards_empty_box_after_clamping(self):
        transform = ZoomTransform(0, 0, 40, 8.0)
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
            (Detection(140, 70, 180, 110, 0.92, 7),),
            ZoomTransform(80, 40, 160, 2.0),
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
                (Detection(100, 80, 220, 300, 0.95, 0),),
                ZoomTransform(80, 40, 160, 2.0),
                AimSettings(confidence=0.35),
            )
        )

    def test_outside_expanded_seed_and_low_confidence_fall_back(self):
        for detection in (
            Detection(300, 10, 319, 29, 0.9, 7),
            Detection(140, 70, 180, 110, 0.1, 7),
        ):
            with self.subTest(detection=detection):
                self.assertIsNone(
                    compose_zoom_refinement(
                        self.base_player(),
                        (detection,),
                        ZoomTransform(80, 40, 160, 2.0),
                        AimSettings(confidence=0.35),
                    )
                )

    def test_association_includes_exact_12px_and_20_percent_margins(self):
        transform = ZoomTransform(80, 0, 160, 2.0)
        exact_boundary = Detection(86, 70, 106, 90, 0.92, 7)
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
            Detection(85.96, 70, 105.96, 90, 0.92, 7),
            Detection(86, 69.96, 106, 89.96, 0.92, 7),
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


if __name__ == "__main__":
    unittest.main()
