import unittest

import numpy as np
import onnxruntime as ort

from jitter_app.ai.detection import (
    ModelContractError,
    OnnxDetector,
    build_letterbox_transform,
    map_detection_to_source,
    model_resource_path,
    parse_output,
    preprocess_frame,
)
from jitter_app.ai.targeting import AimSettings, Detection, analyze_detections
from jitter_app.ai.zoom import ZoomTransform, compose_zoom_refinement
from jitter_app.presentation.overlay import project_overlay_boxes


class NodeArg:
    def __init__(self, name, type_, shape):
        self.name = name
        self.type = type_
        self.shape = shape


class ModelMeta:
    def __init__(self, values=None):
        self.custom_metadata_map = values or {}


class Session:
    def __init__(
        self, inputs=None, outputs=None, providers=None, result=None,
        metadata=None,
    ):
        self._inputs = inputs or [NodeArg("images", "tensor(float)", [1, 3, 320, 320])]
        self._outputs = outputs or [NodeArg("output0", "tensor(float)", [1, 300, 6])]
        self._providers = providers or ["DmlExecutionProvider", "CPUExecutionProvider"]
        self._metadata = ModelMeta(metadata)
        self.result = result if result is not None else valid_output()
        self.run_calls = []

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return self._providers

    def get_modelmeta(self):
        return self._metadata

    def run(self, output_names, inputs):
        self.run_calls.append((output_names, inputs))
        return [self.result]


def valid_output():
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, 0] = (10, 20, 30, 40, 0.75, 7)
    return output


RAW_COUNTS = {160: 525, 320: 2100, 640: 8400}
RAW_METADATA = {"task": "detect", "names": "{0: 'Enemy'}"}


def valid_raw_output(input_size=640):
    output = np.zeros((1, 5, RAW_COUNTS[input_size]), dtype=np.float32)
    output[0, :, 0] = (320, 320, 80, 160, 0.90)
    return output


class DetectionFunctionTests(unittest.TestCase):
    def test_letterbox_plans_landscape_portrait_and_half_up_odd_padding(self):
        landscape = build_letterbox_transform(1920, 1080, 320)
        portrait = build_letterbox_transform(1080, 1920, 320)
        odd = build_letterbox_transform(640, 361, 320)
        one_pixel = build_letterbox_transform(100000, 1, 160)
        ultrawide = build_letterbox_transform(3440, 1440, 320)
        square = build_letterbox_transform(777, 777, 160)

        self.assertEqual(
            (landscape.resized_width, landscape.resized_height,
             landscape.pad_left, landscape.pad_top,
             landscape.pad_right, landscape.pad_bottom),
            (320, 180, 0, 70, 0, 70),
        )
        self.assertEqual(
            (portrait.resized_width, portrait.resized_height,
             portrait.pad_left, portrait.pad_top,
             portrait.pad_right, portrait.pad_bottom),
            (180, 320, 70, 0, 70, 0),
        )
        self.assertEqual(
            (odd.resized_width, odd.resized_height,
             odd.pad_left, odd.pad_top, odd.pad_right, odd.pad_bottom),
            (320, 181, 0, 69, 0, 70),
        )
        self.assertEqual(
            (one_pixel.resized_width, one_pixel.resized_height,
             one_pixel.pad_left, one_pixel.pad_top,
             one_pixel.pad_right, one_pixel.pad_bottom),
            (160, 1, 0, 79, 0, 80),
        )
        self.assertEqual(
            (ultrawide.resized_width, ultrawide.resized_height,
             ultrawide.pad_left, ultrawide.pad_top,
             ultrawide.pad_right, ultrawide.pad_bottom),
            (320, 134, 0, 93, 0, 93),
        )
        self.assertEqual(
            (square.resized_width, square.resized_height,
             square.pad_left, square.pad_top,
             square.pad_right, square.pad_bottom),
            (160, 160, 0, 0, 0, 0),
        )

    def test_preprocess_letterboxes_with_exact_114_padding(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :, 0] = 255

        tensor = preprocess_frame(frame, 320)

        self.assertEqual(tensor.shape, (1, 3, 320, 320))
        np.testing.assert_allclose(tensor[0, :, 0, 0], [114 / 255] * 3)
        np.testing.assert_allclose(tensor[0, :, 70, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(tensor[0, :, 249, 319], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(tensor[0, :, 250, 319], [114 / 255] * 3)

    def test_inverse_letterbox_maps_content_and_rejects_padding_only_boxes(self):
        transform = build_letterbox_transform(1920, 1080, 320)
        mapped = map_detection_to_source(
            Detection(32, 82, 96, 142, 0.9, 7), transform
        )
        self.assertEqual(
            (mapped.x1, mapped.y1, mapped.x2, mapped.y2),
            (192.0, 72.0, 576.0, 432.0),
        )
        self.assertIsNone(map_detection_to_source(
            Detection(20, 0, 100, 69, 0.9, 7), transform
        ))
        crossing = map_detection_to_source(
            Detection(0, 60, 320, 100, 0.9, 7), transform
        )
        self.assertEqual(
            (crossing.x1, crossing.y1, crossing.x2, crossing.y2),
            (0.0, 0.0, 1920.0, 180.0),
        )

    def test_inverse_uses_integer_resized_height_for_odd_padding(self):
        transform = build_letterbox_transform(640, 361, 320)
        mapped = map_detection_to_source(
            Detection(40, 79, 280, 239, 0.9, 7), transform
        )
        np.testing.assert_allclose(
            (mapped.x1, mapped.y1, mapped.x2, mapped.y2),
            (80.0, 19.94475138121547, 560.0, 339.060773480663),
            rtol=0,
            atol=1e-12,
        )

    def test_landscape_inverse_mapping_is_exact_at_every_model_size(self):
        cases = (
            (160, Detection(16, 41, 48, 71, 0.9, 7)),
            (320, Detection(32, 82, 96, 142, 0.9, 7)),
            (640, Detection(64, 164, 192, 284, 0.9, 7)),
        )
        for input_size, model_box in cases:
            with self.subTest(input_size=input_size):
                mapped = map_detection_to_source(
                    model_box,
                    build_letterbox_transform(1920, 1080, input_size),
                )
                self.assertEqual(
                    (mapped.x1, mapped.y1, mapped.x2, mapped.y2),
                    (192.0, 72.0, 576.0, 432.0),
                )

    def test_portrait_inverse_mapping_is_exact_at_every_model_size(self):
        cases = (
            (160, Detection(41, 16, 71, 48, 0.9, 7)),
            (320, Detection(82, 32, 142, 96, 0.9, 7)),
            (640, Detection(164, 64, 284, 192, 0.9, 7)),
        )
        for input_size, model_box in cases:
            with self.subTest(input_size=input_size):
                mapped = map_detection_to_source(
                    model_box,
                    build_letterbox_transform(1080, 1920, input_size),
                )
                self.assertEqual(
                    (mapped.x1, mapped.y1, mapped.x2, mapped.y2),
                    (72.0, 192.0, 432.0, 576.0),
                )

    def test_preprocess_resizes_canonical_frame_to_each_supported_input(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        frame[0, 0] = (255, 128, 0)
        for input_size in (160, 320, 640):
            with self.subTest(input_size=input_size):
                tensor = preprocess_frame(frame, input_size)
                self.assertEqual(tensor.shape, (1, 3, input_size, input_size))
                self.assertEqual(tensor.dtype, np.float32)
                self.assertTrue(tensor.flags.c_contiguous)
                np.testing.assert_allclose(
                    tensor[0, :, 0, 0], [1.0, 128 / 255, 0.0]
                )

    def test_output_coordinates_remain_in_model_space(self):
        cases = (
            (160, (10, 20, 30, 40), (10.0, 20.0, 30.0, 40.0)),
            (320, (10, 20, 30, 40), (10.0, 20.0, 30.0, 40.0)),
            (640, (10, 20, 30, 40), (10.0, 20.0, 30.0, 40.0)),
        )
        for input_size, raw_box, expected in cases:
            with self.subTest(input_size=input_size):
                output = np.zeros((1, 300, 6), dtype=np.float32)
                output[0, 0] = (*raw_box, 0.75, 7)
                detection = parse_output(output, input_size)[0]
                self.assertEqual(
                    (detection.x1, detection.y1, detection.x2, detection.y2),
                    expected,
                )

    def test_output_clips_to_model_space_and_rejects_empty_boxes(self):
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, :2] = (
            (-20, -20, 80, 80, 0.8, 7),
            (-20, 2, -4, 20, 0.9, 0),
        )
        detections = parse_output(output, 640)
        self.assertEqual(len(detections), 1)
        self.assertEqual(
            (detections[0].x1, detections[0].y1,
             detections[0].x2, detections[0].y2),
            (0.0, 0.0, 80.0, 80.0),
        )

    def test_preprocess_returns_normalized_nchw_float32(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        frame[0, 0] = (255, 128, 0)
        tensor = preprocess_frame(frame)
        self.assertEqual(tensor.shape, (1, 3, 320, 320))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0, :, 0, 0], [1.0, 128 / 255, 0.0])

    def test_preprocess_rejects_malformed_frame_shape_and_type(self):
        with self.assertRaisesRegex(ValueError, "RGB"):
            preprocess_frame(np.zeros((320, 320), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "positive"):
            preprocess_frame(np.zeros((0, 320, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "uint8 pixels"):
            preprocess_frame(np.zeros((320, 320, 3), dtype=np.float32))

    def test_parse_output_rejects_malformed_rows(self):
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, :3] = [[1, 2, 10, 20, .8, 7],
                          [5, 5, 4, 6, .9, 0],
                          [np.nan, 1, 2, 3, .9, 7]]
        detections = parse_output(output)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, 7)
        self.assertEqual(
            (detections[0].x1, detections[0].y1,
             detections[0].x2, detections[0].y2),
            (1.0, 2.0, 10.0, 20.0),
        )
        self.assertAlmostEqual(detections[0].confidence, .8, places=6)

    def test_parse_output_clips_before_rejecting_non_positive_boxes(self):
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, :2] = [[-10, -10, 10, 10, .8, 7],
                          [-10, 1, -2, 10, .9, 0]]
        detections = parse_output(output)
        self.assertEqual(len(detections), 1)
        self.assertEqual(
            (detections[0].x1, detections[0].y1,
             detections[0].x2, detections[0].y2),
            (0.0, 0.0, 10.0, 10.0),
        )

    def test_parse_output_requires_exact_model_shape(self):
        with self.assertRaises(ModelContractError):
            parse_output(np.zeros((1, 1, 6), dtype=np.float32))

    def test_model_resource_path_targets_bundled_model(self):
        self.assertEqual(model_resource_path().name, "all_games_320.onnx")
        self.assertEqual(model_resource_path().parent.name, "models")


class OnnxDetectorTests(unittest.TestCase):
    def test_legacy_and_raw_detectors_map_the_same_source_box_at_every_size(self):
        cases = (
            (160, (16, 41, 48, 71), (32, 56, 32, 30)),
            (320, (32, 82, 96, 142), (64, 112, 64, 60)),
            (640, (64, 164, 192, 284), (128, 224, 128, 120)),
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for input_size, legacy_box, raw_candidate in cases:
            with self.subTest(input_size=input_size):
                legacy_output = np.zeros((1, 300, 6), dtype=np.float32)
                legacy_output[0, 0] = (*legacy_box, 0.9, 7)
                raw_output = np.zeros(
                    (1, 5, RAW_COUNTS[input_size]), dtype=np.float32
                )
                raw_output[0, :, 0] = (*raw_candidate, 0.9)
                legacy_session = Session(
                    inputs=[NodeArg(
                        "images", "tensor(float)",
                        [1, 3, input_size, input_size],
                    )],
                    result=legacy_output,
                )
                raw_session = Session(
                    inputs=[NodeArg(
                        "images", "tensor(float)",
                        [1, 3, input_size, input_size],
                    )],
                    outputs=[NodeArg(
                        "output0", "tensor(float)",
                        [1, 5, RAW_COUNTS[input_size]],
                    )],
                    result=raw_output,
                    metadata=RAW_METADATA,
                )
                legacy = OnnxDetector(
                    "legacy.onnx",
                    session_factory=lambda *_args, **_kwargs: legacy_session,
                ).detect(frame)
                raw = OnnxDetector(
                    "raw.onnx",
                    session_factory=lambda *_args, **_kwargs: raw_session,
                ).detect(frame)
                self.assertEqual(len(legacy), 1)
                self.assertEqual(len(raw), 1)
                self.assertEqual(
                    (legacy[0].x1, legacy[0].y1,
                     legacy[0].x2, legacy[0].y2),
                    (192.0, 72.0, 576.0, 432.0),
                )
                self.assertEqual(
                    (raw[0].x1, raw[0].y1, raw[0].x2, raw[0].y2),
                    (192.0, 72.0, 576.0, 432.0),
                )
                self.assertEqual(legacy[0].confidence, raw[0].confidence)
                self.assertAlmostEqual(legacy[0].confidence, 0.9, places=6)
                self.assertEqual(legacy[0].class_id, 7)
                self.assertEqual(raw[0].class_id, 0)
                self.assertEqual(len(legacy_session.run_calls), 1)
                self.assertEqual(len(raw_session.run_calls), 1)

    def test_legacy_and_raw_detectors_reject_padding_only_boxes_at_every_size(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for input_size in (160, 320, 640):
            with self.subTest(input_size=input_size):
                transform = build_letterbox_transform(1920, 1080, input_size)
                legacy_output = np.zeros((1, 300, 6), dtype=np.float32)
                legacy_output[0, 0] = (
                    input_size / 4, 0, input_size * 3 / 4,
                    transform.pad_top, 0.9, 7,
                )
                raw_output = np.zeros(
                    (1, 5, RAW_COUNTS[input_size]), dtype=np.float32
                )
                raw_output[0, :, 0] = (
                    input_size / 2, transform.pad_top / 2,
                    input_size / 2, transform.pad_top, 0.9,
                )
                legacy_session = Session(
                    inputs=[NodeArg(
                        "images", "tensor(float)",
                        [1, 3, input_size, input_size],
                    )],
                    result=legacy_output,
                )
                raw_session = Session(
                    inputs=[NodeArg(
                        "images", "tensor(float)",
                        [1, 3, input_size, input_size],
                    )],
                    outputs=[NodeArg(
                        "output0", "tensor(float)",
                        [1, 5, RAW_COUNTS[input_size]],
                    )],
                    result=raw_output,
                    metadata=RAW_METADATA,
                )
                legacy = OnnxDetector(
                    "legacy.onnx",
                    session_factory=lambda *_args, **_kwargs: legacy_session,
                ).detect(frame)
                raw = OnnxDetector(
                    "raw.onnx",
                    session_factory=lambda *_args, **_kwargs: raw_session,
                ).detect(frame)
                self.assertEqual(legacy, ())
                self.assertEqual(raw, ())

    def test_accepts_exact_raw_contract_for_each_supported_input_size(self):
        for input_size, candidate_count in RAW_COUNTS.items():
            with self.subTest(input_size=input_size):
                session = Session(
                    inputs=[NodeArg(
                        "images", "tensor(float)",
                        [1, 3, input_size, input_size],
                    )],
                    outputs=[NodeArg(
                        "output0", "tensor(float)", [1, 5, candidate_count]
                    )],
                    result=valid_raw_output(input_size),
                    metadata=RAW_METADATA,
                )
                detector = OnnxDetector(
                    "model.onnx",
                    session_factory=lambda *_args, **_kwargs: session,
                )
                self.assertEqual(detector.input_size, input_size)

    def test_raw_detect_routes_to_decoder_and_maps_enemy_to_player(self):
        session = Session(
            inputs=[NodeArg("images", "tensor(float)", [1, 3, 640, 640])],
            outputs=[NodeArg("output0", "tensor(float)", [1, 5, 8400])],
            result=valid_raw_output(640),
            metadata=RAW_METADATA,
        )
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
        self.assertEqual(len(detections), 1)
        self.assertEqual(
            (detections[0].x1, detections[0].y1,
             detections[0].x2, detections[0].y2,
             detections[0].class_id),
            (140.0, 120.0, 180.0, 200.0, 0),
        )
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=6)
        self.assertEqual(session.run_calls[0][1]["images"].shape,
                         (1, 3, 640, 640))

    def test_legacy_contract_does_not_require_ultralytics_metadata(self):
        session = Session(metadata={})
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        self.assertEqual(len(detector.detect(
            np.zeros((320, 320, 3), dtype=np.uint8)
        )), 1)

    def test_raw_single_class_label_is_informational(self):
        for label in ("Enemy", "enemy", "person", "custom target"):
            with self.subTest(label=label):
                session = Session(
                    outputs=[NodeArg(
                        "output0", "tensor(float)", [1, 5, 2100]
                    )],
                    result=valid_raw_output(320),
                    metadata={"task": "detect", "names": repr({0: label})},
                )
                detector = OnnxDetector(
                    "model.onnx",
                    session_factory=lambda *_args, **_kwargs: session,
                )
                self.assertEqual(detector.detect(
                    np.zeros((320, 320, 3), dtype=np.uint8)
                )[0].class_id, 0)

    def test_raw_contract_allows_additional_string_metadata(self):
        session = Session(
            outputs=[NodeArg("output0", "tensor(float)", [1, 5, 2100])],
            result=valid_raw_output(320),
            metadata={
                "task": "detect",
                "names": "{0: 'Enemy'}",
                "author": "Ultralytics",
            },
        )
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        self.assertEqual(len(detector.detect(
            np.zeros((320, 320, 3), dtype=np.uint8)
        )), 1)

    def test_rejects_raw_shape_mismatch_orientation_and_wrong_static_types(self):
        rejected = (
            (160, [1, 5, 2100]),
            (320, [1, 2100, 5]),
            (640, [1, 6, 8400]),
            (640, [1, 5, 8399]),
            (640, [1, 5, 8400.0]),
        )
        for input_size, output_shape in rejected:
            with self.subTest(input_size=input_size, output_shape=output_shape):
                session = Session(
                    inputs=[NodeArg(
                        "images", "tensor(float)",
                        [1, 3, input_size, input_size],
                    )],
                    outputs=[NodeArg(
                        "output0", "tensor(float)", output_shape
                    )],
                    metadata=RAW_METADATA,
                )
                with self.assertRaisesRegex(
                    ModelContractError, "\\[1,300,6\\].*raw one-class"
                ):
                    OnnxDetector(
                        "model.onnx",
                        session_factory=lambda *_args, **_kwargs: session,
                    )

    def test_raw_contract_requires_safe_exact_one_class_detect_metadata(self):
        rejected_metadata = (
            {},
            {"task": "segment", "names": "{0: 'Enemy'}"},
            {"task": "detect", "names": ""},
            {"task": "detect", "names": "not a literal"},
            {"task": "detect", "names": "__import__('os').system('bad')"},
            {"task": "detect", "names": "{False: 'Enemy'}"},
            {"task": "detect", "names": "{1: 'Enemy'}"},
            {"task": "detect", "names": "{0: ''}"},
            {"task": "detect", "names": "{0: 'Enemy', 1: 'Other'}"},
            {"task": "detect", "names": "{0: 'Enemy'}", 1: "x"},
            {"task": "detect", "names": "{0: 'Enemy'}", "author": 1},
        )
        for metadata in rejected_metadata:
            with self.subTest(metadata=metadata):
                session = Session(
                    outputs=[NodeArg(
                        "output0", "tensor(float)", [1, 5, 2100]
                    )],
                    result=valid_raw_output(320),
                    metadata=metadata,
                )
                with self.assertRaisesRegex(
                    ModelContractError, "metadata.*one named class 0"
                ):
                    OnnxDetector(
                        "model.onnx",
                        session_factory=lambda *_args, **_kwargs: session,
                    )

    def test_raw_runtime_output_shape_is_rechecked(self):
        session = Session(
            outputs=[NodeArg("output0", "tensor(float)", [1, 5, 2100])],
            result=np.zeros((1, 2100, 5), dtype=np.float32),
            metadata=RAW_METADATA,
        )
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        with self.assertRaisesRegex(
            ModelContractError, "\\[1,300,6\\].*raw one-class"
        ):
            detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))

    def test_raw_players_flow_to_nearest_target_and_head_hidden_overlay(self):
        output = np.zeros((1, 5, 2100), dtype=np.float32)
        output[0, :, 0] = (40, 40, 20, 40, 0.95)
        output[0, :, 1] = (160, 170, 40, 100, 0.80)
        session = Session(
            outputs=[NodeArg("output0", "tensor(float)", [1, 5, 2100])],
            result=output,
            metadata=RAW_METADATA,
        )
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
        analysis = analyze_detections(
            detections, AimSettings(target_area="head"),
            sequence=1, captured_at=10.0,
        )
        self.assertEqual(analysis.frame.selected_index, 1)
        self.assertEqual(analysis.target.target_class, "player")
        self.assertEqual(
            (analysis.target.aim_x, analysis.target.aim_y), (160.0, 140.0)
        )
        self.assertEqual(
            len(project_overlay_boxes(
                analysis.frame, now=10.0, show_heads=False
            )),
            2,
        )

    def test_raw_player_refinement_preserves_source_geometry(self):
        base_output = np.zeros((1, 5, 2100), dtype=np.float32)
        base_output[0, :, 0] = (160, 170, 40, 100, 0.90)
        refined_output = np.zeros((1, 5, 2100), dtype=np.float32)
        refined_output[0, :, 0] = (160, 160, 40, 160, 0.95)
        session = Session(
            outputs=[NodeArg("output0", "tensor(float)", [1, 5, 2100])],
            result=base_output,
            metadata=RAW_METADATA,
        )
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        base = analyze_detections(
            detector.detect(frame), AimSettings(target_area="head"),
            sequence=4, captured_at=20.0,
        )
        session.result = refined_output
        crop = np.zeros((160, 160, 3), dtype=np.uint8)
        refined = compose_zoom_refinement(
            base,
            detector.detect(crop),
            ZoomTransform(80, 80, 160, 160, 320, 320, 2.0),
            AimSettings(target_area="head"),
        )
        self.assertIsNotNone(refined)
        self.assertEqual(
            (refined.frame.detections[0].x1,
             refined.frame.detections[0].y1,
             refined.frame.detections[0].x2,
             refined.frame.detections[0].y2,
             refined.frame.detections[0].class_id),
            (150.0, 120.0, 170.0, 200.0, 0),
        )
        self.assertEqual(
            (refined.target.frame_width, refined.target.frame_height),
            (320, 320),
        )
        self.assertEqual(
            (refined.frame.frame_width, refined.frame.frame_height),
            (320, 320),
        )

    def test_accepts_only_exact_supported_static_square_input_sizes(self):
        for input_size in (160, 320, 640):
            with self.subTest(input_size=input_size):
                session = Session(inputs=[NodeArg(
                    "images", "tensor(float)",
                    [1, 3, input_size, input_size],
                )])
                detector = OnnxDetector(
                    "model.onnx",
                    session_factory=lambda *_args, **_kwargs: session,
                )
                self.assertEqual(detector.input_size, input_size)

        rejected_shapes = (
            [1, 3, 128, 128],
            [1, 3, 256, 256],
            [1, 3, 160, 320],
            [1, 3, "height", "width"],
            [1, 160, 160, 3],
        )
        for shape in rejected_shapes:
            with self.subTest(shape=shape):
                session = Session(inputs=[NodeArg(
                    "images", "tensor(float)", shape
                )])
                with self.assertRaisesRegex(
                    ModelContractError, "160, 320, or 640"
                ):
                    OnnxDetector(
                        "model.onnx",
                        session_factory=lambda *_args, **_kwargs: session,
                    )

    def test_rejects_type_coercible_non_integer_input_dimensions(self):
        rejected_shapes = (
            [True, 3, 320, 320],
            [1, 3.0, 320, 320],
            [1, 3, 320.0, 320],
            [1, 3, 320, 320.0],
        )
        for shape in rejected_shapes:
            with self.subTest(shape=shape):
                session = Session(inputs=[NodeArg(
                    "images", "tensor(float)", shape
                )])
                with self.assertRaises(ModelContractError):
                    OnnxDetector(
                        "model.onnx",
                        session_factory=lambda *_args, **_kwargs: session,
                    )

    def test_rejects_type_coercible_non_integer_output_dimensions(self):
        rejected_shapes = (
            [True, 300, 6],
            [1, 300.0, 6],
            [1, 300, 6.0],
        )
        for shape in rejected_shapes:
            with self.subTest(shape=shape):
                session = Session(outputs=[NodeArg(
                    "output0", "tensor(float)", shape
                )])
                with self.assertRaises(ModelContractError):
                    OnnxDetector(
                        "model.onnx",
                        session_factory=lambda *_args, **_kwargs: session,
                    )

    def test_detect_uses_validated_size_for_tensor_and_output_mapping(self):
        output = np.zeros((1, 300, 6), dtype=np.float32)
        output[0, 0] = (300, 300, 340, 340, 0.9, 7)
        session = Session(
            inputs=[NodeArg("images", "tensor(float)", [1, 3, 640, 640])],
            result=output,
        )
        detector = OnnxDetector(
            "model.onnx",
            session_factory=lambda *_args, **_kwargs: session,
        )
        detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
        self.assertEqual(session.run_calls[0][1]["images"].shape, (1, 3, 640, 640))
        self.assertEqual(
            (detections[0].x1, detections[0].y1,
             detections[0].x2, detections[0].y2),
            (150.0, 150.0, 170.0, 170.0),
        )

    def test_detector_uses_fixed_contract_and_dml_first(self):
        calls = []
        session = Session()

        def factory(path, **kwargs):
            calls.append((path, kwargs))
            return session

        detector = OnnxDetector("model.onnx", session_factory=factory)
        self.assertEqual(detector.provider, "DmlExecutionProvider")
        self.assertEqual(len(calls), 1)
        path, kwargs = calls[0]
        self.assertEqual(path, "model.onnx")
        self.assertEqual(kwargs["providers"], ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertFalse(kwargs["sess_options"].enable_mem_pattern)
        self.assertEqual(kwargs["sess_options"].execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)

    def test_detector_falls_back_to_cpu_when_dml_construction_fails(self):
        calls = []
        session = Session(providers=["CPUExecutionProvider"])

        def factory(path, **kwargs):
            calls.append(kwargs["providers"])
            if kwargs["providers"][0] == "DmlExecutionProvider":
                raise RuntimeError("DirectML unavailable")
            return session

        detector = OnnxDetector("model.onnx", session_factory=factory)
        self.assertEqual(calls, [
            ["DmlExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ])
        self.assertEqual(detector.provider, "CPUExecutionProvider")

    def test_detector_rejects_contract_mismatches(self):
        sessions = (
            Session(inputs=[NodeArg("input", "tensor(float)", [1, 3, 320, 320])]),
            Session(inputs=[NodeArg("images", "tensor(uint8)", [1, 3, 320, 320])]),
            Session(inputs=[NodeArg("images", "tensor(float)", [1, 320, 320, 3])]),
            Session(outputs=[NodeArg("output", "tensor(float)", [1, 300, 6])]),
            Session(outputs=[NodeArg("output0", "tensor(uint8)", [1, 300, 6])]),
            Session(outputs=[NodeArg("output0", "tensor(float)", [1, 84, 8400])]),
        )
        for session in sessions:
            with self.subTest(session=session):
                with self.assertRaises(ModelContractError):
                    OnnxDetector(
                        "model.onnx",
                        session_factory=lambda *_args, **_kwargs: session,
                    )

    def test_detect_uses_fixed_input_and_output_names(self):
        session = Session()
        detector = OnnxDetector("model.onnx", session_factory=lambda *_args, **_kwargs: session)
        detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
        self.assertEqual(len(detections), 1)
        self.assertEqual(session.run_calls[0][0], ["output0"])
        self.assertEqual(set(session.run_calls[0][1]), {"images"})
        self.assertEqual(session.run_calls[0][1]["images"].shape, (1, 3, 320, 320))


if __name__ == "__main__":
    unittest.main()
