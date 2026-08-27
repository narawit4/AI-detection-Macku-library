import unittest

import numpy as np

from ai_capture import DxcamCapture, centered_region


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
    def test_centered_region(self):
        self.assertEqual(centered_region(1920, 1080), (800, 380, 1120, 700))

    def test_centered_region_rejects_output_smaller_than_capture(self):
        with self.assertRaisesRegex(ValueError, "Primary output is smaller"):
            centered_region(319, 1080)
        with self.assertRaisesRegex(ValueError, "Primary output is smaller"):
            centered_region(1920, 319)

    def test_capture_uses_numpy_rgb_backend_and_owned_frames(self):
        camera = FakeCamera(frame=np.zeros((320, 320, 3), dtype=np.uint8))
        factory = RecordingCameraFactory(camera)
        capture = DxcamCapture(camera_factory=factory, target_fps=165)

        capture.start()
        frame = capture.read()

        self.assertEqual(factory.create_kwargs, {
            "output_idx": 0, "output_color": "RGB",
            "processor_backend": "numpy", "max_buffer_len": 2,
        })
        self.assertEqual(camera.start_kwargs, {
            "region": (800, 380, 1120, 700), "target_fps": 165,
        })
        self.assertEqual(camera.get_latest_frame_kwargs, {"copy": True})
        self.assertTrue(frame.flags.owndata)
        self.assertEqual(frame.shape, (320, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_read_skips_empty_and_malformed_frames(self):
        for frame in (None, np.zeros((320, 320), dtype=np.uint8),
                      np.zeros((320, 320, 4), dtype=np.uint8)):
            with self.subTest(shape=None if frame is None else frame.shape):
                camera = FakeCamera(frame=frame)
                capture = DxcamCapture(camera_factory=RecordingCameraFactory(camera))
                capture.start()
                self.assertIsNone(capture.read())

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
