import hashlib
import unittest

import numpy as np

import jitter_app.ai.resize as image_resize
from jitter_app.ai.resize import resize_rgb_bilinear


class RgbResizeTests(unittest.TestCase):
    def test_reuses_immutable_cached_coordinate_plan(self):
        image_resize._resize_plan.cache_clear()
        first = image_resize._resize_plan(160, 160, 320)
        second = image_resize._resize_plan(160, 160, 320)
        self.assertIs(first, second)
        for values in first:
            with self.assertRaises(ValueError):
                values.flags.writeable = True

    def test_zoom_size_regression_hashes_are_unchanged(self):
        expected_hashes = {
            160: "73fff29a5890f0cdad009470c7ade901489fcc096309af53489d1812cf23e5a5",
            213: "d3876b8a91d71fcd2303a1f5e94c551761191e534b8861c623a908a1c3a4bd32",
        }
        random = np.random.default_rng(20260827)
        for size, expected_hash in expected_hashes.items():
            with self.subTest(size=size):
                source = random.integers(0, 256, (size, size, 3), dtype=np.uint8)
                resized = resize_rgb_bilinear(source)
                self.assertEqual(
                    hashlib.sha256(resized.tobytes()).hexdigest(),
                    expected_hash,
                )

    def test_produces_owned_contiguous_uint8_outputs_for_supported_sizes(self):
        source = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        for output_size in (160, 320, 640):
            with self.subTest(output_size=output_size):
                resized = resize_rgb_bilinear(source, output_size)
                self.assertEqual(resized.shape, (output_size, output_size, 3))
                self.assertEqual(resized.dtype, np.uint8)
                self.assertTrue(resized.flags.c_contiguous)
                self.assertFalse(np.shares_memory(resized, source))

    def test_preserves_hand_derived_center_and_half_up_rounding(self):
        source = np.array(
            [[[0, 0, 0], [100, 100, 100]],
             [[200, 200, 200], [255, 255, 255]]],
            dtype=np.uint8,
        )
        resized = resize_rgb_bilinear(source, 3)
        self.assertEqual(resized[1, 1].tolist(), [139, 139, 139])

        boundary = np.array(
            [[[84, 84, 84], [244, 244, 244]],
             [[7, 7, 7], [191, 191, 191]]],
            dtype=np.uint8,
        )
        self.assertEqual(
            resize_rgb_bilinear(boundary, 7)[1, 1].tolist(),
            [99, 99, 99],
        )

    def test_rejects_malformed_source_and_output_size(self):
        bad_sources = (
            None,
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2, 4), dtype=np.uint8),
            np.zeros((0, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.float32),
        )
        for source in bad_sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "nonempty RGB uint8"):
                    resize_rgb_bilinear(source, 160)
        for output_size in (True, 0, -1, 1.5):
            with self.subTest(output_size=output_size):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    resize_rgb_bilinear(
                        np.zeros((2, 2, 3), dtype=np.uint8), output_size
                    )


if __name__ == "__main__":
    unittest.main()
