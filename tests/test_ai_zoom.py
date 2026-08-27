import unittest

import numpy as np

from ai_targeting import Detection, TargetSnapshot
from ai_zoom import (
    ZoomTransform,
    build_zoom_input,
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


if __name__ == "__main__":
    unittest.main()
