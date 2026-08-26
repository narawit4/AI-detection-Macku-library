import unittest

import numpy as np
import onnxruntime as ort

from ai_detection import (
    ModelContractError,
    OnnxDetector,
    model_resource_path,
    parse_output,
    preprocess_frame,
)


class NodeArg:
    def __init__(self, name, type_, shape):
        self.name = name
        self.type = type_
        self.shape = shape


class Session:
    def __init__(self, inputs=None, outputs=None, providers=None, result=None):
        self._inputs = inputs or [NodeArg("images", "tensor(float)", [1, 3, 320, 320])]
        self._outputs = outputs or [NodeArg("output0", "tensor(float)", [1, 300, 6])]
        self._providers = providers or ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.result = result if result is not None else valid_output()
        self.run_calls = []

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return self._providers

    def run(self, output_names, inputs):
        self.run_calls.append((output_names, inputs))
        return [self.result]


def valid_output():
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, 0] = (10, 20, 30, 40, 0.75, 7)
    return output


class DetectionFunctionTests(unittest.TestCase):
    def test_preprocess_returns_normalized_nchw_float32(self):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        frame[0, 0] = (255, 128, 0)
        tensor = preprocess_frame(frame)
        self.assertEqual(tensor.shape, (1, 3, 320, 320))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0, :, 0, 0], [1.0, 128 / 255, 0.0])

    def test_preprocess_rejects_wrong_frame_shape_and_type(self):
        with self.assertRaisesRegex(ValueError, "RGB 320x320x3"):
            preprocess_frame(np.zeros((320, 320), dtype=np.uint8))
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
