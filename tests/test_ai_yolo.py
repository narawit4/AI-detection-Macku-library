import unittest

import numpy as np

from ai_targeting import Detection
from ai_yolo import (
    MAX_DETECTIONS,
    MIN_CONFIDENCE,
    NMS_IOU_THRESHOLD,
    RAW_CANDIDATE_COUNTS,
    _nms_keep_positions,
    decode_single_class_yolo,
)


def raw_output(input_size: int) -> np.ndarray:
    return np.zeros(
        (1, 5, RAW_CANDIDATE_COUNTS[input_size]), dtype=np.float32
    )


def put_candidate(
    output: np.ndarray,
    index: int,
    cx: float,
    cy: float,
    width: float,
    height: float,
    confidence: float,
) -> None:
    output[0, :, index] = (cx, cy, width, height, confidence)


class RawYoloDecoderTests(unittest.TestCase):
    def test_contract_constants_are_exact(self):
        self.assertEqual(MIN_CONFIDENCE, 0.05)
        self.assertEqual(NMS_IOU_THRESHOLD, 0.45)
        self.assertEqual(MAX_DETECTIONS, 300)
        self.assertEqual(
            RAW_CANDIDATE_COUNTS, {160: 525, 320: 2100, 640: 8400}
        )

    def test_center_boxes_scale_to_canonical_space_for_every_input_size(self):
        for input_size in (160, 320, 640):
            with self.subTest(input_size=input_size):
                output = raw_output(input_size)
                put_candidate(output, 0, input_size / 2, input_size / 2,
                              input_size / 4, input_size / 2, 0.80)
                detection = decode_single_class_yolo(output, input_size)[0]
                self.assertEqual(
                    (detection.x1, detection.y1, detection.x2, detection.y2,
                     detection.class_id),
                    (120.0, 80.0, 200.0, 240.0, 0),
                )
                self.assertAlmostEqual(detection.confidence, 0.8, places=6)

    def test_coordinates_clip_after_scaling_and_collapsed_boxes_are_removed(self):
        for input_size in (160, 320, 640):
            with self.subTest(input_size=input_size):
                output = raw_output(input_size)
                put_candidate(
                    output, 0,
                    input_size * 0.05, input_size * 0.05,
                    input_size * 0.20, input_size * 0.20, 0.90,
                )
                put_candidate(
                    output, 1,
                    -input_size * 0.10, input_size * 0.05,
                    input_size * 0.05, input_size * 0.05, 0.95,
                )
                detections = decode_single_class_yolo(output, input_size)
                self.assertEqual(len(detections), 1)
                np.testing.assert_allclose(
                    (detections[0].x1, detections[0].y1,
                     detections[0].x2, detections[0].y2),
                    (0.0, 0.0, 48.0, 48.0),
                    atol=1e-5,
                )

    def test_invalid_rows_are_skipped_without_mutating_output(self):
        output = raw_output(320)
        candidates = (
            (100, 100, 20, 20, 0.80),
            (100, 100, 20, 20, 0.049),
            (100, 100, 20, 20, 1.01),
            (100, 100, 0, 20, 0.90),
            (100, 100, -1, 20, 0.90),
            (np.nan, 100, 20, 20, 0.90),
            (100, np.inf, 20, 20, 0.90),
        )
        for index, values in enumerate(candidates):
            put_candidate(output, index, *values)
        before = output.copy()
        detections = decode_single_class_yolo(output, 320)
        np.testing.assert_array_equal(output, before)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, 0)

    def test_exact_confidence_floor_is_inclusive(self):
        output = raw_output(160)
        put_candidate(output, 0, 20, 20, 10, 10, 0.05)
        self.assertEqual(len(decode_single_class_yolo(output, 160)), 1)
        self.assertEqual(decode_single_class_yolo(raw_output(160), 160), ())

    def test_rejects_wrong_shape_orientation_size_and_nonnumeric_array(self):
        invalid = (
            (np.zeros((1, 5, 524), dtype=np.float32), 160),
            (np.zeros((1, 525, 5), dtype=np.float32), 160),
            (np.zeros((1, 5, 525), dtype=np.float32), 320),
            (np.zeros((1, 5, 525), dtype="U1"), 160),
            (np.zeros((1, 5, 525), dtype=np.bool_), 160),
        )
        for output, input_size in invalid:
            with self.subTest(shape=output.shape, input_size=input_size):
                with self.assertRaises(ValueError):
                    decode_single_class_yolo(output, input_size)
        with self.assertRaises(ValueError):
            decode_single_class_yolo([[[]]], 160)
        with self.assertRaises(ValueError):
            decode_single_class_yolo(raw_output(160), 128)
        with self.assertRaises(ValueError):
            decode_single_class_yolo(raw_output(160), 160.0)
        with self.assertRaises(ValueError):
            decode_single_class_yolo(raw_output(160), True)

    def test_nms_suppresses_only_iou_strictly_greater_than_threshold(self):
        boxes = np.asarray(
            [[0, 0, 29, 10], [11, 0, 40, 10], [10, 0, 39, 10]],
            dtype=np.float64,
        )
        confidences = np.asarray([0.9, 0.8, 0.7], dtype=np.float64)
        raw_indices = np.asarray([0, 1, 2], dtype=np.int64)
        kept = _nms_keep_positions(boxes, confidences, raw_indices)
        np.testing.assert_array_equal(kept, [0, 1])

    def test_equal_confidence_nms_and_final_output_preserve_detector_order(self):
        output = raw_output(320)
        put_candidate(output, 8, 100, 100, 40, 40, 0.90)
        put_candidate(output, 3, 102, 102, 40, 40, 0.90)
        put_candidate(output, 5, 250, 250, 20, 20, 0.95)
        detections = decode_single_class_yolo(output, 320)
        self.assertEqual(
            [(item.x1, item.y1) for item in detections],
            [(82.0, 82.0), (240.0, 240.0)],
        )
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=6)
        self.assertAlmostEqual(detections[1].confidence, 0.95, places=6)

    def test_decoder_stops_after_three_hundred_nonoverlapping_survivors(self):
        output = raw_output(160)
        for index in range(301):
            put_candidate(
                output, index, 0.1 + index * 0.4, 1.0, 0.1, 0.1, 0.90
            )
        detections = decode_single_class_yolo(output, 160)
        self.assertEqual(len(detections), MAX_DETECTIONS)
        self.assertAlmostEqual(detections[0].x1, 0.1, places=6)
        self.assertLess(detections[-1].x1, 240.0)

    def test_private_nms_interface_uses_one_to_many_inputs_not_pairwise_iou(self):
        boxes = np.asarray([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float64)
        kept = _nms_keep_positions(
            boxes,
            np.asarray([0.8, 0.7], dtype=np.float64),
            np.asarray([4, 9], dtype=np.int64),
        )
        self.assertEqual(boxes.shape, (2, 4))
        np.testing.assert_array_equal(kept, [0, 1])
