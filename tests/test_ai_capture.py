import unittest

import numpy as np

from jitter_app.ai.capture import DxcamCapture, full_output_region


class FakeCamera:
    def __init__(self, width=1920, height=1080, frame=None):
        self.width = width
        self.height = height
        self.frame = frame
        self.start_kwargs = None
        self.started = False
        self.stop_calls = 0
        self.release_calls = 0

    def start(self, **kwargs):
        self.start_kwargs = kwargs
        self.started = True

    def get_latest_frame(self, **kwargs):
        self.get_latest_frame_kwargs = kwargs
        return self.frame

    def stop(self):
        self.stop_calls += 1

    def release(self):
        self.release_calls += 1


class RecordingCameraFactory:
    def __init__(self, camera):
        self.camera = camera
        self.create_kwargs = None

    def __call__(self, **kwargs):
        self.create_kwargs = kwargs
        return self.camera


class CaptureTests(unittest.TestCase):
    def test_full_output_region_requires_positive_integer_geometry(self):
        self.assertEqual(full_output_region(1920, 1080), (0, 0, 1920, 1080))
        for width, height in (
            (0, 1080),
            (1920, 0),
            (True, 1080),
            (1920, 1.5),
        ):
            with self.subTest(width=width, height=height):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    full_output_region(width, height)

    def test_capture_requests_full_output_and_returns_owned_native_rgb(self):
        source = np.zeros((1080, 1920, 3), dtype=np.uint8)
        camera = FakeCamera(width=1920, height=1080, frame=source)
        factory = RecordingCameraFactory(camera)
        capture = DxcamCapture(camera_factory=factory, target_fps=165)

        capture.start()
        frame = capture.read()

        self.assertEqual(factory.create_kwargs, {
            "output_idx": 0, "output_color": "RGB",
            "processor_backend": "numpy", "max_buffer_len": 2,
        })
        self.assertEqual(camera.start_kwargs, {
            "region": (0, 0, 1920, 1080), "target_fps": 165,
        })
        self.assertEqual(camera.get_latest_frame_kwargs, {"copy": True})
        self.assertTrue(frame.flags.owndata)
        self.assertTrue(frame.flags.c_contiguous)
        self.assertFalse(np.shares_memory(frame, source))
        self.assertEqual(frame.shape, (1080, 1920, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_read_preserves_none_as_no_new_frame(self):
        camera = FakeCamera(frame=None)
        capture = DxcamCapture(camera_factory=RecordingCameraFactory(camera))
        capture.start()

        self.assertIsNone(capture.read())

    def test_read_rejects_every_non_none_malformed_frame(self):
        malformed_frames = (
            object(),
            np.zeros((320, 320), dtype=np.uint8),
            np.zeros((320, 320, 4), dtype=np.uint8),
            np.zeros((0, 320, 3), dtype=np.uint8),
            np.zeros((320, 0, 3), dtype=np.uint8),
            np.zeros((320, 320, 3), dtype=np.float32),
            np.zeros((320, 320, 3), dtype=np.bool_),
        )
        for frame in malformed_frames:
            with self.subTest(
                frame_type=type(frame),
                shape=getattr(frame, "shape", None),
            ):
                camera = FakeCamera(frame=frame)
                capture = DxcamCapture(camera_factory=RecordingCameraFactory(camera))
                capture.start()
                with self.assertRaisesRegex(
                    ValueError,
                    "^AI capture frame must be nonempty RGB uint8$",
                ):
                    capture.read()

    def test_close_stops_and_releases_once_when_started(self):
        camera = FakeCamera(frame=np.zeros((320, 320, 3), dtype=np.uint8))
        capture = DxcamCapture(camera_factory=RecordingCameraFactory(camera))
        capture.start()

        capture.close()
        capture.close()

        self.assertEqual(camera.stop_calls, 1)
        self.assertEqual(camera.release_calls, 1)

    def test_close_releases_even_when_stop_raises(self):
        class FailingStopCamera(FakeCamera):
            def stop(self):
                self.stop_calls += 1
                raise RuntimeError("stop failed")

        camera = FailingStopCamera()
        capture = DxcamCapture(camera_factory=RecordingCameraFactory(camera))
        capture.start()

        capture.close()

        self.assertEqual(camera.stop_calls, 1)
        self.assertEqual(camera.release_calls, 1)


if __name__ == "__main__":
    unittest.main()
