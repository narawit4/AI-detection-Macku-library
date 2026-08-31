import threading
import time
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from jitter_app.ai.service import AiEvent, AiService
from jitter_app.ai.targeting import (
    AimSettings,
    Detection,
    DetectionFrameSnapshot,
    analyze_detections as real_analyze_detections,
)
from jitter_app.motion.combined import CombinedMotionEngine, MotionSources
from jitter_app.motion.engine import MotionSettings


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    wake = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return True
        wake.wait(0.005)
    return predicate()


def rgb_frame(value=0, *, width=320, height=320):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[0, 0] = value
    return frame


class FakeCapture:
    def __init__(self, frames=None):
        self.frames = list(
            frames if frames is not None else [rgb_frame(10)]
        )
        self.closed = threading.Event()

    def start(self):
        pass

    def read(self):
        if self.frames:
            return self.frames.pop(0)
        return None

    def close(self):
        self.closed.set()


class ControlledCapture(FakeCapture):
    def __init__(self, frames):
        super().__init__(frames)
        self._permits = threading.Semaphore(0)

    def release_frame(self):
        self._permits.release()

    def read(self):
        if not self.frames:
            return None
        if not self._permits.acquire(timeout=0.01):
            return None
        return super().read()


class CountingCapture(FakeCapture):
    def __init__(self, frames=None, start_error=None, read_error=None):
        super().__init__(frames)
        self.start_error = start_error
        self.read_error = read_error
        self.start_calls = 0
        self.read_calls = 0
        self.close_calls = 0

    def start(self):
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error

    def read(self):
        self.read_calls += 1
        if self.read_error is not None:
            raise self.read_error
        return super().read()

    def close(self):
        self.close_calls += 1
        super().close()


class FakeDetector:
    provider = "DmlExecutionProvider"

    def __init__(self):
        self.calls = []

    def detect(self, frame):
        self.calls.append(frame)
        value = float(frame[0, 0, 0])
        return (
            Detection(value, value, value + 10, value + 10, 0.9, 7),
        )


class SequenceDetector:
    provider = "CPUExecutionProvider"

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def detect(self, _frame):
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return ()


class SequentialDetector:
    provider = "DmlExecutionProvider"

    def __init__(self, outputs, second_call_hook=None):
        self.outputs = list(outputs)
        self.frames = []
        self.second_call_hook = second_call_hook

    def detect(self, frame):
        self.frames.append(frame)
        if len(self.frames) == 2 and self.second_call_hook is not None:
            self.second_call_hook()
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class BlockingDetector:
    provider = "CPUExecutionProvider"

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def detect(self, _frame):
        self.entered.set()
        self.release.wait(1.0)
        return (Detection(150, 150, 170, 170, 0.9, 7),)


class FakeClock:
    def __init__(self, values):
        self._values = iter(values)
        self._last = 0.0

    def __call__(self):
        try:
            self._last = next(self._values)
        except StopIteration:
            pass
        return self._last


class MutableClock:
    def __init__(self, value=0.0):
        self._value = value
        self._lock = threading.Lock()

    def set(self, value):
        with self._lock:
            self._value = value

    def __call__(self):
        with self._lock:
            return self._value


class ObservedRLock:
    def __init__(self):
        self._lock = threading.RLock()
        self._guard = threading.Lock()
        self._attempts = 0

    def __enter__(self):
        with self._guard:
            self._attempts += 1
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()

    @property
    def attempts(self):
        with self._guard:
            return self._attempts

    def reset_attempts(self):
        with self._guard:
            self._attempts = 0

    def acquire(self):
        with self._guard:
            self._attempts += 1
        return self._lock.acquire()

    def release(self):
        self._lock.release()


class SelectiveGateLock:
    def __init__(self, gated_thread_name):
        self._lock = threading.RLock()
        self._guard = threading.Lock()
        self._gated_thread_name = gated_thread_name
        self._gate_used = False
        self.gate_entered = threading.Event()
        self.release_gate = threading.Event()

    def __enter__(self):
        should_gate = False
        with self._guard:
            if (
                threading.current_thread().name == self._gated_thread_name
                and not self._gate_used
            ):
                self._gate_used = True
                should_gate = True
        if should_gate:
            self.gate_entered.set()
            self.release_gate.wait(1.0)
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()


class PublicationObserverLock:
    """Expose the first publication release after a checked worker frame."""

    def __init__(self, worker_name):
        self._lock = threading.RLock()
        self._guard = threading.Lock()
        self._worker_name = worker_name
        self._phase = "awaiting_currentness_check"
        self.currentness_check_released = threading.Event()
        self.allow_publication = threading.Event()
        self.first_publication_released = threading.Event()
        self.release_observation = threading.Event()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()
        event = None
        release = None
        with self._guard:
            if threading.current_thread().name == self._worker_name:
                if self._phase == "awaiting_currentness_check":
                    self._phase = "awaiting_first_publication"
                    event = self.currentness_check_released
                    release = self.allow_publication
                elif self._phase == "awaiting_first_publication":
                    self._phase = "observing"
                    event = self.first_publication_released
                    release = self.release_observation
        if event is not None:
            event.set()
            release.wait(1.0)


class AiServiceTests(unittest.TestCase):
    def test_default_capture_factory_receives_valid_capture_fps(self):
        target_fps_values = []

        class RecordingCapture:
            def __init__(self, *, target_fps):
                target_fps_values.append(target_fps)

        with mock.patch("jitter_app.ai.service.DxcamCapture", RecordingCapture):
            service = AiService(lambda _event: None, capture_fps=165)
            self.addCleanup(service.close)
            capture = service._capture_factory()

        self.assertIsInstance(capture, RecordingCapture)
        self.assertEqual(target_fps_values, [165])

    def test_invalid_capture_fps_falls_back_to_120(self):
        invalid_values = (0, -1, 165.0, "165", True, float("nan"), object())
        for invalid in invalid_values:
            with self.subTest(capture_fps=invalid):
                target_fps_values = []

                class RecordingCapture:
                    def __init__(self, *, target_fps):
                        target_fps_values.append(target_fps)

                with mock.patch("jitter_app.ai.service.DxcamCapture", RecordingCapture):
                    service = AiService(lambda _event: None, capture_fps=invalid)
                    try:
                        service._capture_factory()
                    finally:
                        service.close()

                self.assertEqual(target_fps_values, [120])

    def test_explicit_capture_factory_remains_authoritative(self):
        capture = object()

        class FalseyCaptureFactory:
            def __bool__(self):
                return False

            def __call__(self):
                return capture

        service = AiService(
            lambda _event: None,
            capture_factory=FalseyCaptureFactory(),
            capture_fps=165,
        )
        self.addCleanup(service.close)

        self.assertIs(service._capture_factory(), capture)

    def make_zoom_service(
        self,
        detector,
        *,
        frames=None,
        capture=None,
        clock=time.perf_counter,
    ):
        source_frames = (
            list(frames)
            if frames is not None
            else [np.zeros((320, 320, 3), dtype=np.uint8)]
        )
        if capture is None:
            capture = FakeCapture(source_frames)
        events = []
        service = AiService(
            events.append,
            detector_factory=lambda _path: detector,
            capture_factory=lambda: capture,
            clock=clock,
        )
        self.addCleanup(service.close)
        return service, events

    def small_head(self, x=160.0):
        return Detection(x - 9.0, 150.0, x + 9.0, 168.0, 0.9, 7)

    def large_head(self, x):
        return Detection(x - 20.0, 120.0, x + 20.0, 200.0, 0.9, 7)

    def tracking_head(self, x, y=100.0):
        return Detection(x - 5.0, y - 5.0, x + 5.0, y + 5.0, 0.9, 7)

    def release_and_wait(self, capture, service, sequence):
        capture.release_frame()
        self.assertTrue(wait_until(
            lambda: service.latest_detection_snapshot() is not None
            and service.latest_detection_snapshot().sequence == sequence
        ))

    def test_normal_gate_publishes_nearest_head_or_player_each_frame(self):
        near_player = Detection(120.0, 132.0, 200.0, 272.0, 0.9, 0)
        far_player = Detection(0.0, 132.0, 80.0, 272.0, 0.9, 0)
        far_head = self.large_head(40.0)
        near_head = self.large_head(160.0)
        detections = ((far_head, near_player), (far_player, near_head))
        detector = SequentialDetector(detections)
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in detections
        ])
        clock = MutableClock(10.0)
        service, _events = self.make_zoom_service(
            detector,
            capture=capture,
            clock=clock,
        )
        service.start(AimSettings, lambda: True)

        self.release_and_wait(capture, service, 1)
        first = service.latest_snapshot()
        self.assertEqual(first.target_class, "player")
        self.assertEqual((first.aim_x, first.aim_y), (160.0, 160.0))
        self.assertEqual(service.latest_detection_snapshot().selected_index, 1)

        clock.set(10.01)
        self.release_and_wait(capture, service, 2)
        second = service.latest_snapshot()
        self.assertEqual(second.target_class, "head")
        self.assertEqual((second.aim_x, second.aim_y), (160.0, 160.0))
        self.assertEqual(service.latest_detection_snapshot().selected_index, 1)

    def test_normal_gate_selects_nearest_box_without_ambiguity_hold(self):
        outputs = (
            (self.large_head(100.0),),
            (self.large_head(100.0),),
            (self.large_head(99.0), self.large_head(101.0)),
            (self.large_head(102.0),),
            (self.large_head(103.0),),
        )
        detector = SequentialDetector(outputs)
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in outputs
        ])
        clock = MutableClock(10.0)
        service, _events = self.make_zoom_service(
            detector,
            capture=capture,
            clock=clock,
        )
        service.start(AimSettings, lambda: True)

        for sequence in (1, 2):
            clock.set(10.0 + sequence / 100.0)
            self.release_and_wait(capture, service, sequence)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 100.0)

        clock.set(10.03)
        self.release_and_wait(capture, service, 3)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 101.0)
        selected = service.latest_detection_snapshot()
        self.assertEqual(selected.selected_index, 1)
        self.assertEqual(selected.detections, outputs[2])

        clock.set(10.04)
        self.release_and_wait(capture, service, 4)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 102.0)
        current = service.latest_detection_snapshot()
        self.assertEqual(current.selected_index, 0)
        self.assertEqual(current.detections, outputs[3])

        clock.set(10.05)
        self.release_and_wait(capture, service, 5)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 103.0)
        self.assertEqual(service.latest_detection_snapshot().selected_index, 0)

    def test_normal_gate_replacement_publishes_first_current_observation(self):
        original = (self.large_head(100.0),)
        replacement = (self.large_head(250.0),)
        outputs = (
            original,
            original,
            replacement,
            replacement,
            replacement,
        )
        detector = SequentialDetector(outputs)
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in outputs
        ])
        clock = MutableClock(10.0)
        service, _events = self.make_zoom_service(
            detector,
            capture=capture,
            clock=clock,
        )
        service.start(AimSettings, lambda: True)

        self.release_and_wait(capture, service, 1)
        clock.set(10.01)
        self.release_and_wait(capture, service, 2)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 100.0)

        for sequence, captured_at in ((3, 10.160), (4, 10.177)):
            clock.set(captured_at)
            self.release_and_wait(capture, service, sequence)
            self.assertAlmostEqual(service.latest_snapshot().aim_x, 250.0)
            current = service.latest_detection_snapshot()
            self.assertEqual(current.selected_index, 0)
            self.assertEqual(current.detections, replacement)

        clock.set(10.194)
        self.release_and_wait(capture, service, 5)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 250.0)
        self.assertEqual(service.latest_detection_snapshot().selected_index, 0)

    def test_normal_gate_missing_frame_clears_only_that_frame_target(self):
        outputs = (
            (self.large_head(100.0),),
            (self.large_head(100.0),),
            (),
            (self.large_head(101.0),),
            (self.large_head(102.0),),
        )
        detector = SequentialDetector(outputs)
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in outputs
        ])
        clock = MutableClock(10.0)
        service, _events = self.make_zoom_service(
            detector,
            capture=capture,
            clock=clock,
        )
        service.start(AimSettings, lambda: True)

        for sequence in (1, 2):
            clock.set(10.0 + sequence / 100.0)
            self.release_and_wait(capture, service, sequence)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 100.0)

        clock.set(10.03)
        self.release_and_wait(capture, service, 3)
        self.assertIsNone(service.latest_snapshot())

        clock.set(10.04)
        self.release_and_wait(capture, service, 4)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 101.0)

        clock.set(10.05)
        self.release_and_wait(capture, service, 5)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 102.0)

    def test_target_area_change_publishes_new_current_frame_aim_point(self):
        player = Detection(140.0, 40.0, 180.0, 180.0, 0.9, 0)
        detector = SequentialDetector(((player,),) * 4)
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(4)
        ])
        clock = MutableClock(10.0)
        service, _events = self.make_zoom_service(
            detector,
            capture=capture,
            clock=clock,
        )
        current = {"settings": AimSettings(target_area="head")}
        service.start(lambda: current["settings"], lambda: True)

        self.release_and_wait(capture, service, 1)
        self.assertAlmostEqual(service.latest_snapshot().aim_y, 68.0)
        clock.set(10.01)
        self.release_and_wait(capture, service, 2)
        self.assertAlmostEqual(service.latest_snapshot().aim_y, 68.0)

        current["settings"] = AimSettings(target_area="upper_body")
        clock.set(10.02)
        self.release_and_wait(capture, service, 3)
        self.assertAlmostEqual(service.latest_snapshot().aim_y, 82.0)
        clock.set(10.03)
        self.release_and_wait(capture, service, 4)
        self.assertAlmostEqual(service.latest_snapshot().aim_y, 82.0)

    def test_rapid_round_trip_target_area_change_publishes_current_setting(self):
        player = Detection(140.0, 40.0, 180.0, 180.0, 0.9, 0)
        detector = SequentialDetector(((player,),) * 4)
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(4)
        ])
        clock = MutableClock(10.0)
        service, _events = self.make_zoom_service(
            detector,
            capture=capture,
            clock=clock,
        )
        current = {"settings": AimSettings(target_area="head")}
        service.start(lambda: current["settings"], lambda: True)
        self.release_and_wait(capture, service, 1)
        clock.set(10.01)
        self.release_and_wait(capture, service, 2)
        self.assertIsNotNone(service.latest_snapshot())

        current["settings"] = AimSettings(target_area="chest")
        service.reset_targeting()
        current["settings"] = AimSettings(target_area="head")
        service.reset_targeting()
        clock.set(10.02)
        self.release_and_wait(capture, service, 3)

        self.assertIsNotNone(service.latest_snapshot())
        clock.set(10.03)
        self.release_and_wait(capture, service, 4)
        self.assertIsNotNone(service.latest_snapshot())

    def test_reset_targeting_immediately_clears_movement_but_keeps_overlay(self):
        player = Detection(140.0, 40.0, 180.0, 180.0, 0.9, 0)
        detector = SequentialDetector(((player,),))
        capture = ControlledCapture([
            rgb_frame(width=640, height=360)
        ])
        service, _events = self.make_zoom_service(detector, capture=capture)
        service.start(AimSettings, lambda: False)
        self.release_and_wait(capture, service, 1)
        overlay_before = service.latest_detection_snapshot()
        self.assertIsNotNone(service.latest_snapshot())

        service.reset_targeting()

        self.assertIsNone(service.latest_snapshot())
        overlay_after = service.latest_detection_snapshot()
        self.assertEqual(overlay_after.detections, overlay_before.detections)
        self.assertIsNone(overlay_after.selected_index)
        self.assertEqual(
            (overlay_after.frame_width, overlay_after.frame_height),
            (640, 360),
        )

    def test_reset_targeting_preserves_explicit_capture_viewport(self):
        service = AiService(lambda _event: None)
        self.addCleanup(service.close)
        detection = Detection(150, 150, 170, 170, 0.9, 7)
        service._latest_detection = DetectionFrameSnapshot(
            4,
            2.0,
            (detection,),
            0,
            320,
            320,
            1920,
            1080,
            800,
            380,
        )

        service.reset_targeting()

        self.assertEqual(
            service.latest_detection_snapshot(),
            DetectionFrameSnapshot(
                4,
                2.0,
                (detection,),
                None,
                320,
                320,
                1920,
                1080,
                800,
                380,
            ),
        )

    def test_reset_targeting_rejects_an_in_flight_old_setting_publication(self):
        player = Detection(140.0, 40.0, 180.0, 180.0, 0.9, 0)
        detector = SequentialDetector(((player,), (player,)))
        capture = ControlledCapture([
            rgb_frame(width=640, height=360),
            rgb_frame(width=800, height=600),
        ])
        provider_entered = threading.Event()
        release_provider = threading.Event()
        calls = 0

        def settings_provider():
            nonlocal calls
            calls += 1
            if calls == 2:
                provider_entered.set()
                release_provider.wait(1.0)
            return AimSettings(target_area="head")

        def analyze_with_viewport(detections, settings, **kwargs):
            return real_analyze_detections(
                detections,
                settings,
                **kwargs,
                output_width=1920,
                output_height=1080,
                capture_left=800,
                capture_top=380,
            )

        service, _events = self.make_zoom_service(detector, capture=capture)
        with mock.patch(
            "jitter_app.ai.service.analyze_detections",
            side_effect=analyze_with_viewport,
        ):
            service.start(settings_provider, lambda: False)
            self.release_and_wait(capture, service, 1)
            capture.release_frame()
            self.assertTrue(provider_entered.wait(1.0))

            service.reset_targeting()
            release_provider.set()

            self.assertTrue(wait_until(
                lambda: service.latest_detection_snapshot() is not None
                and service.latest_detection_snapshot().sequence == 2
            ))
        self.assertIsNone(service.latest_snapshot())
        latest = service.latest_detection_snapshot()
        self.assertIsNone(latest.selected_index)
        self.assertEqual(
            (latest.frame_width, latest.frame_height),
            (800, 600),
        )
        self.assertEqual(
            (
                latest.output_width,
                latest.output_height,
                latest.capture_left,
                latest.capture_top,
            ),
            (1920, 1080, 800, 380),
        )

    def test_crossing_targets_select_nearest_current_box(self):
        person_a = (130.0, 140.0, 150.0, 160.0, 175.0, 190.0)
        person_b = (None, 220.0, 190.0, 160.0, 145.0, 130.0)
        outputs = []
        for a_x, b_x in zip(person_a, person_b):
            boxes = [self.tracking_head(a_x)]
            if b_x is not None:
                boxes.insert(0, self.tracking_head(b_x))
            outputs.append(tuple(boxes))
        detector = SequentialDetector(outputs)
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in outputs
        ])
        clock = MutableClock(10.0)
        service, _events = self.make_zoom_service(
            detector,
            capture=capture,
            clock=clock,
        )
        service.start(AimSettings, lambda: False)

        for sequence in (1, 2, 3):
            clock.set(10.0 + sequence / 60.0)
            self.release_and_wait(capture, service, sequence)
            self.assertAlmostEqual(
                service.latest_snapshot().aim_x,
                person_a[sequence - 1],
            )

        clock.set(10.0 + 4 / 60.0)
        self.release_and_wait(capture, service, 4)

        self.assertAlmostEqual(service.latest_snapshot().aim_x, 160.0)
        current = service.latest_detection_snapshot()
        self.assertEqual(current.selected_index, 0)
        self.assertEqual(len(current.detections), 2)

        clock.set(10.0 + 5 / 60.0)
        self.release_and_wait(capture, service, 5)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 145.0)
        self.assertEqual(service.latest_detection_snapshot().selected_index, 0)

        clock.set(10.0 + 6 / 60.0)
        self.release_and_wait(capture, service, 6)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 130.0)
        self.assertEqual(service.latest_detection_snapshot().selected_index, 0)

    def test_service_replacement_publishes_first_current_frame(self):
        original = (self.small_head(100.0),)
        replacement = (self.small_head(220.0),)
        detector = SequentialDetector((
            original,
            replacement,
            replacement,
            replacement,
        ))
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(4)
        ])
        clock = MutableClock(10.0)
        service, _events = self.make_zoom_service(
            detector,
            capture=capture,
            clock=clock,
        )
        service.start(AimSettings, lambda: False)

        self.release_and_wait(capture, service, 1)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 100.0)

        for sequence, captured_at in ((2, 10.150), (3, 10.167)):
            clock.set(captured_at)
            self.release_and_wait(capture, service, sequence)
            self.assertAlmostEqual(service.latest_snapshot().aim_x, 220.0)
            current = service.latest_detection_snapshot()
            self.assertEqual(current.selected_index, 0)
            self.assertEqual(current.detections, replacement)

        clock.set(10.184)
        self.release_and_wait(capture, service, 4)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 220.0)
        self.assertEqual(service.latest_detection_snapshot().selected_index, 0)

    def test_current_frame_replacement_combines_with_jitter_immediately(self):
        original = (self.small_head(100.0),)
        replacement = (self.small_head(220.0),)
        detector = SequentialDetector((
            original,
            replacement,
        ))
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(2)
        ])
        clock = MutableClock(10.0)
        service, _events = self.make_zoom_service(
            detector,
            capture=capture,
            clock=clock,
        )
        service.start(AimSettings, lambda: False)

        self.release_and_wait(capture, service, 1)
        clock.set(10.150)
        self.release_and_wait(capture, service, 2)
        current = service.latest_detection_snapshot()
        self.assertEqual(current.detections, replacement)
        self.assertEqual(current.selected_index, 0)

        class FixedJitter:
            def step(self, _settings, _dt, _elapsed):
                return 3, -2

        class RecordingAim:
            def __init__(self):
                self.targets = []

            def step(self, target, _settings, _now):
                self.targets.append(target)
                return (0, 0) if target is None else (40, 40)

        aim = RecordingAim()
        engine = CombinedMotionEngine(
            MotionSources(jitter=True, ai=True),
            jitter_engine_factory=FixedJitter,
            aim_engine_factory=lambda: aim,
        )

        report = engine.step(
            MotionSettings(),
            service.latest_snapshot(),
            AimSettings(),
            dt=0.01,
            elapsed=0.02,
            now=10.151,
        )

        self.assertEqual(aim.targets, [service.latest_snapshot()])
        self.assertEqual(report, (43, 38))

    def test_old_generation_cannot_publish_tracker_recovery(self):
        tracking_head = self.tracking_head

        class RecoveryBlockingDetector:
            provider = "DmlExecutionProvider"

            def __init__(self):
                self.calls = 0
                self.entered = threading.Event()
                self.release = threading.Event()

            def detect(self, _frame):
                self.calls += 1
                outputs = (
                    (tracking_head(100.0),),
                    (tracking_head(99.0), tracking_head(101.0)),
                    (tracking_head(102.0),),
                    (tracking_head(103.0),),
                )
                if self.calls == 4:
                    self.entered.set()
                    self.release.wait(1.0)
                return outputs[self.calls - 1]

        old_detector = RecoveryBlockingDetector()
        new_detector = SequentialDetector(((self.small_head(35.0),),))
        detectors = iter((old_detector, new_detector))
        old_capture = FakeCapture([rgb_frame() for _ in range(4)])
        new_capture = FakeCapture([rgb_frame()])
        captures = iter((old_capture, new_capture))
        service = AiService(
            lambda _event: None,
            detector_factory=lambda _path: next(detectors),
            capture_factory=lambda: next(captures),
        )
        self.addCleanup(old_detector.release.set)
        self.addCleanup(service.close)

        service.start(AimSettings, lambda: False)
        self.assertTrue(old_detector.entered.wait(1.0))
        service.stop("restart")
        service.start(AimSettings, lambda: False)
        self.assertTrue(wait_until(
            lambda: service.latest_snapshot() is not None
            and service.latest_snapshot().aim_x == 35.0
        ))

        old_detector.release.set()

        self.assertTrue(old_capture.closed.wait(1.0))
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 35.0)
        self.assertEqual(
            service.latest_detection_snapshot().detections,
            (self.small_head(35.0),),
        )

    def test_refinement_replaces_only_confirmed_base_box_and_stays_out_of_tracker(self):
        selected = self.small_head(160.0)
        unrelated = Detection(20.0, 40.0, 40.0, 140.0, 0.9, 0)
        refined = Detection(112.0, 100.0, 124.0, 112.0, 0.95, 7)
        next_base = self.small_head(115.0)
        detector = SequentialDetector((
            (selected, unrelated),
            (refined,),
            (next_base,),
        ))
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(2)
        ])
        gate = {"active": True}
        clock = MutableClock(10.0)
        service, _events = self.make_zoom_service(
            detector,
            capture=capture,
            clock=clock,
        )
        service.start(AimSettings, lambda: gate["active"])

        self.release_and_wait(capture, service, 1)
        published = service.latest_detection_snapshot()
        self.assertEqual(len(published.detections), 2)
        self.assertNotEqual(published.detections[0], selected)
        self.assertEqual(published.detections[1], unrelated)
        self.assertEqual(published.selected_index, 0)
        self.assertEqual(
            [frame.shape for frame in detector.frames],
            [(320, 320, 3), (213, 213, 3)],
        )

        gate["active"] = False
        clock.set(10.01)
        self.release_and_wait(capture, service, 2)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 115.0)
        self.assertEqual(service.latest_detection_snapshot().detections, (next_base,))

    def test_stop_after_capture_prevents_base_detector_call(self):
        detector = BlockingDetector()
        stopped = threading.Event()
        capture = FakeCapture([np.zeros((320, 320, 3), dtype=np.uint8)])
        service = AiService(
            lambda _event: None,
            detector_factory=lambda _path: detector,
            capture_factory=lambda: capture,
        )
        self.addCleanup(service.close)
        self.addCleanup(detector.release.set)

        def stop_before_base_inference():
            service.stop("settings race")
            stopped.set()
            return AimSettings()

        service.start(stop_before_base_inference)

        self.assertTrue(stopped.wait(1.0))
        self.assertFalse(detector.entered.wait(0.1))

    def test_zoom_gate_false_publishes_base_with_one_inference(self):
        detector = SequentialDetector(((Detection(140, 80, 180, 160, 0.9, 0),),))
        service, events = self.make_zoom_service(detector)
        generation = service.start(AimSettings, lambda: False)
        self.assertTrue(wait_until(
            lambda: len(detector.frames) == 1
            and service.latest_snapshot() is not None
        ))
        self.assertIsNotNone(generation)
        self.assertEqual(len(detector.frames), 1)
        self.assertEqual(service.latest_snapshot().target_class, "player")
        self.assertFalse(any(
            event.kind == "zoom" and event.payload == 1.5
            for event in events
        ))

    def test_ineligible_large_target_uses_one_inference(self):
        detector = SequentialDetector(
            ((Detection(140, 80, 180, 193, 0.9, 0),),)
        )
        service, events = self.make_zoom_service(detector)
        service.start(AimSettings, lambda: True)
        self.assertTrue(wait_until(
            lambda: len(detector.frames) == 1
            and service.latest_detection_snapshot() is not None
        ))
        self.assertEqual(len(detector.frames), 1)
        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(service.latest_detection_snapshot().selected_index, 0)
        self.assertFalse(any(event.kind == "zoom" for event in events))

    def test_refinement_miss_publishes_same_frame_base_fallback(self):
        base = (Detection(140, 80, 180, 160, 0.9, 0),)
        detector = SequentialDetector((base, ()))
        service, events = self.make_zoom_service(detector)
        service.start(AimSettings, lambda: True)
        self.assertTrue(wait_until(
            lambda: len(detector.frames) == 2
            and service.latest_detection_snapshot() is not None
        ))
        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(service.latest_detection_snapshot().detections, base)
        self.assertFalse(any(event.kind == "zoom" for event in events))

    def test_first_small_target_uses_one_half_x_and_publishes_movement(self):
        base = (self.small_head(),)
        refined = (Detection(97, 97, 115, 115, 0.93, 7),)
        detector = SequentialDetector((base, refined))
        capture = ControlledCapture(
            [np.zeros((320, 320, 3), dtype=np.uint8)]
        )
        clock = MutableClock(10.0)
        service, events = self.make_zoom_service(
            detector, capture=capture, clock=clock
        )
        service.start(AimSettings, lambda: True)

        self.release_and_wait(capture, service, 1)

        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(service.latest_detection_snapshot().selected_index, 0)
        self.assertEqual(
            [event.payload for event in events if event.kind == "zoom"],
            [1.5],
        )
        self.assertEqual(len(detector.frames), 2)
        self.assertEqual(
            [frame.shape for frame in detector.frames],
            [(320, 320, 3), (213, 213, 3)],
        )

    def test_stable_target_enters_two_x_then_recoil_downgrades(self):
        base = (self.small_head(),)
        jumped = (self.small_head(178.01),)
        one_half_refined = (Detection(97, 97, 115, 115, 0.93, 7),)
        two_x_refined = (Detection(71, 71, 89, 89, 0.93, 7),)
        detector = SequentialDetector((
            base, one_half_refined,
            base, one_half_refined,
            base, two_x_refined,
            jumped, one_half_refined,
        ))
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(4)
        ])
        clock = MutableClock(10.0)
        service, events = self.make_zoom_service(
            detector, capture=capture, clock=clock
        )
        service.start(AimSettings, lambda: True)

        self.release_and_wait(capture, service, 1)
        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(len(detector.frames), 2)

        clock.set(10.05)
        self.release_and_wait(capture, service, 2)
        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(len(detector.frames), 4)

        clock.set(10.1)
        self.release_and_wait(capture, service, 3)
        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(len(detector.frames), 6)

        clock.set(10.11)
        self.release_and_wait(capture, service, 4)
        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(len(detector.frames), 8)
        self.assertEqual(
            [event.payload for event in events if event.kind == "zoom"],
            [1.5, 2.0, 1.5],
        )
        self.assertEqual(
            [frame.shape for frame in detector.frames],
            [
                (320, 320, 3),
                (213, 213, 3),
                (320, 320, 3),
                (213, 213, 3),
                (320, 320, 3),
                (160, 160, 3),
                (320, 320, 3),
                (213, 213, 3),
            ],
        )

    def test_native_zoom_uses_per_frame_geometry_for_both_refinements(self):
        base = (Detection(302, 162, 338, 198, 0.90, 7),)
        refined_one_half = (Detection(195, 102, 231, 138, 0.93, 7),)
        refined_two = (Detection(142, 72, 178, 108, 0.94, 7),)
        detector = SequentialDetector((
            base,
            refined_one_half,
            base,
            refined_two,
        ))
        capture = ControlledCapture([
            rgb_frame(width=640, height=360),
            rgb_frame(width=640, height=360),
        ])
        clock = MutableClock(10.0)
        service, events = self.make_zoom_service(
            detector,
            capture=capture,
            clock=clock,
        )
        service.start(AimSettings, lambda: True)

        self.release_and_wait(capture, service, 1)
        clock.set(10.11)
        self.release_and_wait(capture, service, 2)

        self.assertEqual(
            [frame.shape for frame in detector.frames],
            [
                (360, 640, 3),
                (240, 427, 3),
                (360, 640, 3),
                (180, 320, 3),
            ],
        )
        self.assertEqual(
            [event.payload for event in events if event.kind == "zoom"],
            [1.5, 2.0],
        )

    def test_refinement_miss_resets_confirmation_without_holding_old_target(self):
        base = (self.small_head(),)
        refined = (Detection(97, 97, 115, 115, 0.93, 7),)
        detector = SequentialDetector((
            base, refined,
            base, refined,
            base, (),
            base, refined,
            base, refined,
        ))
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(5)
        ])
        clock = MutableClock(10.0)
        service, events = self.make_zoom_service(
            detector, capture=capture, clock=clock
        )
        service.start(AimSettings, lambda: True)

        self.release_and_wait(capture, service, 1)
        clock.set(10.05)
        self.release_and_wait(capture, service, 2)
        self.assertIsNotNone(service.latest_snapshot())

        clock.set(10.1)
        self.release_and_wait(capture, service, 3)
        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(service.latest_detection_snapshot().detections, base)

        clock.set(10.11)
        self.release_and_wait(capture, service, 4)
        self.assertIsNotNone(service.latest_snapshot())

        clock.set(10.12)
        self.release_and_wait(capture, service, 5)
        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(len(detector.frames), 10)
        self.assertEqual(
            [event.payload for event in events if event.kind == "zoom"],
            [1.5, 1.0, 1.5],
        )
        self.assertEqual(
            [frame.shape for frame in detector.frames],
            [
                (320, 320, 3),
                (213, 213, 3),
                (320, 320, 3),
                (213, 213, 3),
                (320, 320, 3),
                (160, 160, 3),
                (320, 320, 3),
                (213, 213, 3),
                (320, 320, 3),
                (213, 213, 3),
            ],
        )

    def test_false_gate_resets_zoom_stability_and_uses_nearest_base_selection(self):
        associated = Detection(80, 80, 120, 160, 0.9, 0)
        centered = Detection(140, 80, 180, 160, 0.9, 0)
        detector = SequentialDetector((
            (associated,), (),
            (associated, centered),
            (associated, centered), (),
        ))
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(3)
        ])
        gate = {"active": True}
        clock = MutableClock(10.0)
        service, _events = self.make_zoom_service(
            detector, capture=capture, clock=clock
        )
        service.start(AimSettings, lambda: gate["active"])

        self.release_and_wait(capture, service, 1)
        self.assertIsNotNone(service.latest_snapshot())

        gate["active"] = False
        clock.set(10.01)
        self.release_and_wait(capture, service, 2)
        self.assertAlmostEqual(service.latest_snapshot().aim_x, 160.0)
        self.assertEqual(service.latest_detection_snapshot().selected_index, 1)

        gate["active"] = True
        clock.set(10.02)
        self.release_and_wait(capture, service, 3)
        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(service.latest_detection_snapshot().selected_index, 1)
        self.assertEqual(len(detector.frames), 5)

    def test_restart_uses_fresh_stability_state(self):
        base = (self.small_head(),)
        refined = (Detection(142, 142, 178, 178, 0.93, 7),)
        old_detector = SequentialDetector((base, refined, base, refined))
        new_detector = SequentialDetector((base, refined))
        old_capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(2)
        ])
        new_capture = ControlledCapture(
            [np.zeros((320, 320, 3), dtype=np.uint8)]
        )
        detectors = iter((old_detector, new_detector))
        captures = iter((old_capture, new_capture))
        clock = MutableClock(10.0)
        service = AiService(
            lambda _event: None,
            detector_factory=lambda _path: next(detectors),
            capture_factory=lambda: next(captures),
            clock=clock,
        )
        self.addCleanup(service.close)

        service.start(AimSettings, lambda: True)
        self.release_and_wait(old_capture, service, 1)
        clock.set(10.05)
        self.release_and_wait(old_capture, service, 2)
        self.assertIsNotNone(service.latest_snapshot())

        service.stop("restart")
        self.assertIsNone(service.latest_snapshot())
        clock.set(20.0)
        service.start(AimSettings, lambda: True)
        self.release_and_wait(new_capture, service, 1)
        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(len(new_detector.frames), 2)

    def test_gate_release_during_second_call_discards_refinement(self):
        gate = {"active": True}
        detector = SequentialDetector(
            (
                (Detection(140, 80, 180, 160, 0.9, 0),),
                (Detection(144, 135, 174, 165, 0.95, 7),),
            ),
            second_call_hook=lambda: gate.update(active=False),
        )
        service, events = self.make_zoom_service(detector)
        service.start(AimSettings, lambda: gate["active"])
        self.assertTrue(wait_until(
            lambda: len(detector.frames) == 2
            and service.latest_snapshot() is not None
        ))
        self.assertEqual(service.latest_snapshot().target_class, "player")
        self.assertFalse(any(
            event.kind == "zoom" and event.payload == 1.5
            for event in events
        ))

    def test_restart_during_second_call_cannot_publish_old_refinement(self):
        new_detection = Detection(30, 30, 40, 40, 0.9, 7)
        new_detector = SequentialDetector(((new_detection,),))
        service_holder = {}

        def restart_service():
            service_holder["service"].stop("restart")
            service_holder["service"].start(AimSettings, lambda: False)

        old_detector = SequentialDetector(
            (
                (Detection(140, 80, 180, 160, 0.9, 0),),
                (Detection(144, 135, 174, 165, 0.95, 7),),
            ),
            second_call_hook=restart_service,
        )
        detectors = iter((old_detector, new_detector))
        captures = iter((
            FakeCapture([np.zeros((320, 320, 3), dtype=np.uint8)]),
            FakeCapture([np.zeros((320, 320, 3), dtype=np.uint8)]),
        ))
        events = []
        service = AiService(
            events.append,
            detector_factory=lambda _path: next(detectors),
            capture_factory=lambda: next(captures),
        )
        service_holder["service"] = service
        self.addCleanup(service.close)

        service.start(AimSettings, lambda: True)
        self.assertTrue(wait_until(
            lambda: service.latest_detection_snapshot() is not None
            and service.latest_detection_snapshot().detections == (new_detection,)
        ))
        self.assertFalse(any(
            event.kind == "zoom" and event.payload == 1.5
            for event in events
        ))

    def test_zoom_events_emit_only_on_success_and_factor_transition(self):
        base = (Detection(140, 80, 180, 160, 0.9, 0),)
        refined = (Detection(96, 90, 116, 110, 0.92, 7),)
        detector = SequentialDetector((base, refined, base, refined, base, ()))
        frames = [np.zeros((320, 320, 3), dtype=np.uint8) for _ in range(3)]
        service, events = self.make_zoom_service(detector, frames=frames)
        service.start(AimSettings, lambda: True)
        self.assertTrue(wait_until(
            lambda: len(detector.frames) == 6
            and service.latest_detection_snapshot() is not None
            and service.latest_detection_snapshot().sequence == 3
            and any(
                event.kind == "zoom" and event.payload == 1.0
                for event in events
            )
        ))
        self.assertEqual(
            [event.payload for event in events if event.kind == "zoom"],
            [1.5, 1.0],
        )
        self.assertEqual(
            [event.targeting_revision for event in events if event.kind == "zoom"],
            [0, 0],
        )
        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(
            [frame.shape for frame in detector.frames],
            [
                (320, 320, 3),
                (213, 213, 3),
                (320, 320, 3),
                (213, 213, 3),
                (320, 320, 3),
                (213, 213, 3),
            ],
        )

    def test_fps_counts_published_frames_not_detector_calls(self):
        base = (Detection(140, 80, 180, 160, 0.9, 0),)
        refined = (Detection(144, 135, 174, 165, 0.92, 7),)
        detector = SequentialDetector((base, refined, base, refined))
        frames = [np.zeros((320, 320, 3), dtype=np.uint8) for _ in range(2)]
        service, events = self.make_zoom_service(
            detector,
            frames=frames,
            clock=FakeClock([0.0, 0.1, 0.2, 0.21, 1.2]),
        )
        service.start(AimSettings, lambda: True)
        self.assertTrue(wait_until(
            lambda: any(event.kind == "fps" for event in events)
        ))
        fps = next(event.payload for event in events if event.kind == "fps")
        self.assertAlmostEqual(fps, 2 / 1.2)
        self.assertEqual(len(detector.frames), 4)

    def test_first_refinement_error_disables_zoom_once_for_generation(self):
        base = (Detection(140, 80, 180, 160, 0.9, 0),)
        refined = (Detection(96, 90, 116, 110, 0.92, 7),)
        detector = SequentialDetector(
            (base, refined, base, RuntimeError("refine failed"), base)
        )
        frames = [np.zeros((320, 320, 3), dtype=np.uint8) for _ in range(3)]
        service, events = self.make_zoom_service(detector, frames=frames)

        with self.assertLogs("jitter_app.ai.service", level="ERROR") as logs:
            service.start(AimSettings, lambda: True)
            self.assertTrue(wait_until(
                lambda: len(detector.frames) == 5
                and service.latest_snapshot() is not None
                and service.latest_snapshot().sequence == 3
                and any(
                    event.kind == "zoom" and event.payload == 1.0
                    for event in events
                )
            ))

        matching_logs = [
            line for line in logs.output
            if "Adaptive AI zoom disabled" in line
        ]
        self.assertEqual(len(matching_logs), 1)
        self.assertEqual(service.latest_snapshot().target_class, "player")
        self.assertFalse(any(event.kind == "error" for event in events))
        self.assertEqual(
            [event.payload for event in events if event.kind == "zoom"],
            [1.5, 1.0],
        )
        self.assertEqual(len(detector.frames), 5)
        self.assertEqual(
            [frame.shape for frame in detector.frames],
            [
                (320, 320, 3),
                (213, 213, 3),
                (320, 320, 3),
                (213, 213, 3),
                (320, 320, 3),
            ],
        )

    def test_refinement_error_keeps_stability_gate_active_after_rearm(self):
        base = (self.small_head(),)
        refined = (Detection(142, 142, 178, 178, 0.93, 7),)
        detector = SequentialDetector((
            base,
            refined,
            base,
            RuntimeError("refine failed"),
            base,
            base,
        ))
        capture = ControlledCapture([
            np.zeros((320, 320, 3), dtype=np.uint8)
            for _ in range(4)
        ])
        gate = {"active": True}
        clock = MutableClock(10.0)
        service, events = self.make_zoom_service(
            detector,
            capture=capture,
            clock=clock,
        )

        with self.assertLogs("jitter_app.ai.service", level="ERROR") as logs:
            service.start(AimSettings, lambda: gate["active"])
            self.release_and_wait(capture, service, 1)
            self.assertIsNotNone(service.latest_snapshot())

            clock.set(10.05)
            self.release_and_wait(capture, service, 2)
            self.assertIsNotNone(service.latest_snapshot())

            gate["active"] = False
            clock.set(10.06)
            self.release_and_wait(capture, service, 3)
            self.assertIsNotNone(service.latest_snapshot())

            gate["active"] = True
            clock.set(10.07)
            self.release_and_wait(capture, service, 4)

        self.assertIsNotNone(service.latest_snapshot())
        self.assertEqual(service.latest_detection_snapshot().selected_index, 0)
        self.assertEqual(len(detector.frames), 6)
        self.assertFalse(any(event.kind == "error" for event in events))
        self.assertEqual(
            sum("Adaptive AI zoom disabled" in line for line in logs.output),
            1,
        )

    def test_reentrant_loading_stop_or_close_does_not_launch_worker(self):
        for action in ("stop", "close"):
            with self.subTest(action=action):
                detector_calls = []
                thread_instances = []

                class InlineThread:
                    def __init__(self, *, target, args, name, daemon):
                        self.target = target
                        self.args = args
                        thread_instances.append(self)

                    def start(self):
                        self.target(*self.args)

                service = None

                def sink(event):
                    if event.kind == "loading":
                        if action == "stop":
                            service.stop("reentrant")
                        else:
                            service.close()

                service = AiService(
                    sink,
                    detector_factory=lambda _path: detector_calls.append(True),
                    capture_factory=FakeCapture,
                )
                with mock.patch("jitter_app.ai.service.threading.Thread", InlineThread):
                    service.start(AimSettings)

                self.assertEqual(thread_instances, [])
                self.assertEqual(detector_calls, [])
                self.assertFalse(service.running)
                self.assertEqual(service.status, "stopped")
                service.close()

    def test_cancelled_worker_checks_before_model_or_detector_construction(self):
        real_thread = threading.Thread
        worker_finished = threading.Event()
        model_calls = []
        detector_calls = []

        class CapturedThread:
            instance = None

            def __init__(self, *, target, args, name, daemon):
                self.target = target
                self.args = args
                self.name = name
                self.daemon = daemon
                CapturedThread.instance = self

            def start(self):
                pass

        service = AiService(
            lambda _event: None,
            detector_factory=lambda _path: detector_calls.append(True),
            capture_factory=FakeCapture,
        )
        self.addCleanup(service.close)
        with mock.patch(
            "jitter_app.ai.service.model_resource_path",
            side_effect=lambda: model_calls.append(True),
        ):
            with mock.patch("jitter_app.ai.service.threading.Thread", CapturedThread):
                service.start(AimSettings)

            worker = CapturedThread.instance
            gated_lock = ObservedRLock()
            service._lock = gated_lock
            gated_lock.acquire()
            gated_lock.reset_attempts()

            def run():
                try:
                    worker.target(*worker.args)
                finally:
                    worker_finished.set()

            try:
                real_thread(
                    target=run,
                    name=worker.name,
                    daemon=worker.daemon,
                ).start()
                self.assertTrue(wait_until(lambda: gated_lock.attempts >= 1))
                worker.args[1].set()
            finally:
                gated_lock.release()
            self.assertTrue(worker_finished.wait(1.0))

        self.assertEqual(model_calls, [])
        self.assertEqual(detector_calls, [])

    def test_thread_start_failure_rolls_back_and_emits_safe_error(self):
        events = []

        class FailingStartThread:
            def __init__(self, **_kwargs):
                pass

            def start(self):
                raise RuntimeError("sensitive scheduler diagnostic")

        service = AiService(
            events.append,
            detector_factory=lambda _path: self.fail("worker must not run"),
            capture_factory=FakeCapture,
        )
        self.addCleanup(service.close)
        with (
            mock.patch("jitter_app.ai.service.threading.Thread", FailingStartThread),
            self.assertLogs("jitter_app.ai.service", level="ERROR") as logs,
        ):
            generation = service.start(AimSettings)

        self.assertIsNone(generation)
        self.assertFalse(service.running)
        self.assertEqual(service.status, "error")
        self.assertIsNone(service.provider)
        self.assertIsNone(service.latest_snapshot())
        self.assertIsNone(service.latest_detection_snapshot())
        self.assertEqual(
            [event for event in events if event.kind == "error"],
            [AiEvent("error", "RuntimeError: AI service failed")],
        )
        self.assertNotIn("sensitive scheduler diagnostic", events[-1].payload)
        self.assertIn("sensitive scheduler diagnostic", "\n".join(logs.output))

    def test_concurrent_start_stop_or_close_signals_new_stop_event(self):
        for action in ("stop", "close"):
            with self.subTest(action=action):
                detector = BlockingDetector()
                capture = CountingCapture()
                caller_returned = threading.Event()
                caller_name = f"AI-{action}-caller"
                gated_lock = SelectiveGateLock(caller_name)
                service = AiService(
                    lambda _event: None,
                    detector_factory=lambda _path: detector,
                    capture_factory=lambda: capture,
                )
                service._lock = gated_lock

                def invoke_lifecycle():
                    if action == "stop":
                        service.stop("concurrent")
                    else:
                        service.close()
                    caller_returned.set()

                caller = threading.Thread(
                    target=invoke_lifecycle,
                    name=caller_name,
                    daemon=True,
                )
                caller.start()
                try:
                    self.assertTrue(gated_lock.gate_entered.wait(1.0))
                    service.start(AimSettings)
                    self.assertTrue(detector.entered.wait(1.0))
                    current_stop_event = service._stop_event

                    gated_lock.release_gate.set()

                    self.assertTrue(caller_returned.wait(1.0))
                    self.assertTrue(current_stop_event.is_set())
                finally:
                    gated_lock.release_gate.set()
                    detector.release.set()
                    service.close()

    def test_ready_worker_publishes_only_latest_snapshot(self):
        events = []
        service = AiService(
            events.append,
            detector_factory=lambda path: FakeDetector(),
            capture_factory=lambda: FakeCapture(
                frames=[rgb_frame(10), rgb_frame(20)]
            ),
            clock=FakeClock([1.0, 1.01, 1.02, 1.03, 1.04]),
        )
        self.addCleanup(service.close)

        service.start(AimSettings)

        self.assertTrue(
            wait_until(
                lambda: service.latest_snapshot() is not None
                and service.latest_snapshot().sequence == 2
            )
        )
        self.assertEqual(service.latest_snapshot().sequence, 2)
        self.assertIn(AiEvent("ready", "DmlExecutionProvider"), events)

    def test_worker_atomically_publishes_target_and_detection_frame(self):
        head = Detection(150, 150, 170, 170, 0.9, 7)
        service = AiService(
            lambda _event: None,
            detector_factory=lambda _path: SequenceDetector([(head,)]),
            capture_factory=lambda: FakeCapture([rgb_frame()]),
        )
        self.addCleanup(service.close)

        service.start(AimSettings)

        self.assertTrue(wait_until(
            lambda: service.latest_detection_snapshot() is not None
        ))
        target = service.latest_snapshot()
        frame = service.latest_detection_snapshot()
        self.assertEqual(target.sequence, frame.sequence)
        self.assertEqual(frame.detections, (head,))
        self.assertEqual(frame.selected_index, 0)

    def test_native_frame_dimensions_publish_atomically_with_target_and_boxes(self):
        frame = rgb_frame(width=640, height=360)
        detector = SequenceDetector([
            (Detection(300, 160, 340, 220, 0.9, 7),),
        ])
        service = AiService(
            detector_factory=lambda _path: detector,
            capture_factory=lambda: FakeCapture([frame]),
            event_sink=lambda _event: None,
        )
        self.addCleanup(service.close)

        service.start(lambda: AimSettings(), lambda: False)
        self.assertTrue(wait_until(lambda: service.latest_snapshot() is not None))

        target = service.latest_snapshot()
        boxes = service.latest_detection_snapshot()
        self.assertEqual((target.frame_width, target.frame_height), (640, 360))
        self.assertEqual((boxes.frame_width, boxes.frame_height), (640, 360))
        self.assertEqual((target.aim_x, target.aim_y), (320.0, 190.0))
        self.assertEqual(detector.calls, 1)

    def test_each_published_sequence_uses_its_own_native_frame_dimensions(self):
        detector = SequentialDetector((
            (Detection(300, 160, 340, 220, 0.9, 7),),
            (Detection(380, 270, 420, 330, 0.9, 7),),
        ))
        capture = ControlledCapture([
            rgb_frame(width=640, height=360),
            rgb_frame(width=800, height=600),
        ])
        service = AiService(
            detector_factory=lambda _path: detector,
            capture_factory=lambda: capture,
            event_sink=lambda _event: None,
        )
        self.addCleanup(service.close)
        service.start(lambda: AimSettings(), lambda: False)

        self.release_and_wait(capture, service, 1)
        first_target = service.latest_snapshot()
        first_boxes = service.latest_detection_snapshot()
        self.assertEqual(
            (first_target.frame_width, first_target.frame_height),
            (640, 360),
        )
        self.assertEqual(
            (first_boxes.frame_width, first_boxes.frame_height),
            (640, 360),
        )

        self.release_and_wait(capture, service, 2)
        second_target = service.latest_snapshot()
        second_boxes = service.latest_detection_snapshot()
        self.assertEqual(
            (second_target.frame_width, second_target.frame_height),
            (800, 600),
        )
        self.assertEqual(
            (second_boxes.frame_width, second_boxes.frame_height),
            (800, 600),
        )
        self.assertEqual(
            [frame.shape for frame in detector.frames],
            [(360, 640, 3), (600, 800, 3)],
        )

    def test_worker_never_exposes_target_before_its_detection_frame(self):
        detector = BlockingDetector()
        service = AiService(
            lambda _event: None,
            detector_factory=lambda _path: detector,
            capture_factory=lambda: FakeCapture([rgb_frame()]),
        )
        self.addCleanup(service.close)
        generation = service.start(AimSettings)
        self.assertTrue(detector.entered.wait(1.0))

        gate_lock = PublicationObserverLock(f"AiInference-{generation}")
        service._lock = gate_lock
        self.addCleanup(gate_lock.release_observation.set)
        self.addCleanup(gate_lock.allow_publication.set)
        self.addCleanup(detector.release.set)
        detector.release.set()

        self.assertTrue(gate_lock.currentness_check_released.wait(1.0))
        gate_lock.allow_publication.set()
        self.assertTrue(gate_lock.first_publication_released.wait(1.0))
        target = service.latest_snapshot()
        frame = service.latest_detection_snapshot()
        self.assertIsNotNone(target)
        self.assertIsNotNone(frame)
        self.assertEqual(target.sequence, frame.sequence)
        self.assertEqual(frame.selected_index, 0)

    def test_stop_invalidates_late_inference_result(self):
        detector = BlockingDetector()
        service = AiService(
            lambda event: None,
            detector_factory=lambda path: detector,
            capture_factory=FakeCapture,
        )
        self.addCleanup(service.close)
        service.start(AimSettings)
        self.assertTrue(detector.entered.wait(1.0))

        service.stop("manual")
        detector.release.set()

        self.assertTrue(wait_until(lambda: not service.running))
        self.assertIsNone(service.latest_snapshot())
        self.assertIsNone(service.latest_detection_snapshot())

    def test_public_state_tracks_provider_and_stop(self):
        ready = threading.Event()
        events = []

        def sink(event):
            events.append(event)
            if event.kind == "ready":
                ready.set()

        service = AiService(
            sink,
            detector_factory=lambda _path: FakeDetector(),
            capture_factory=FakeCapture,
        )
        self.addCleanup(service.close)
        self.assertEqual(service.status, "stopped")
        self.assertIsNone(service.provider)
        self.assertFalse(service.running)

        generation = service.start(AimSettings)
        self.assertTrue(ready.wait(1.0))

        self.assertEqual(service.status, "ready")
        self.assertEqual(service.provider, "DmlExecutionProvider")
        self.assertTrue(service.running)
        with self.assertRaises(AttributeError):
            service.status = "changed"
        with self.assertRaises(AttributeError):
            service.provider = "changed"
        with self.assertRaises(AttributeError):
            service.running = False

        service.stop("disabled")

        self.assertEqual(service.status, "stopped")
        self.assertIsNone(service.provider)
        self.assertFalse(service.running)
        self.assertIn(AiEvent("stopped", "disabled"), events)
        self.assertIsInstance(generation, int)

    def test_explicit_model_path_is_created_on_daemon_worker(self):
        factory_entered = threading.Event()
        release_factory = threading.Event()
        observed = {}

        def detector_factory(path):
            thread = threading.current_thread()
            observed.update(path=path, name=thread.name, daemon=thread.daemon)
            factory_entered.set()
            release_factory.wait(1.0)
            return FakeDetector()

        service = AiService(
            lambda _event: None,
            model_path="chosen.onnx",
            detector_factory=detector_factory,
            capture_factory=FakeCapture,
        )
        self.addCleanup(release_factory.set)
        self.addCleanup(service.close)

        first = service.start(AimSettings)
        second = service.start(AimSettings)

        self.assertTrue(factory_entered.wait(1.0))
        self.assertEqual(second, first)
        self.assertEqual(observed, {
            "path": "chosen.onnx",
            "name": f"AiInference-{first}",
            "daemon": True,
        })
        service.stop()
        release_factory.set()

    def test_start_model_path_overrides_constructor_default_for_one_generation(
        self,
    ):
        observed = []
        ready = threading.Event()

        def detector_factory(path):
            observed.append(path)
            return FakeDetector()

        service = AiService(
            lambda event: ready.set() if event.kind == "ready" else None,
            model_path="constructor.onnx",
            detector_factory=detector_factory,
            capture_factory=FakeCapture,
        )
        self.addCleanup(service.close)
        service.start(AimSettings, model_path=Path("generation.onnx"))

        self.assertTrue(ready.wait(1.0))
        self.assertEqual(observed, [Path("generation.onnx")])

    def test_running_generation_keeps_first_path_and_restart_uses_second_path(
        self,
    ):
        paths = []
        first_path_consumed = threading.Event()
        ready_count = threading.Event()

        def detector_factory(path):
            paths.append(path)
            if len(paths) == 1:
                first_path_consumed.set()
            if len(paths) == 2:
                ready_count.set()
            return FakeDetector()

        service = AiService(
            lambda _event: None,
            detector_factory=detector_factory,
            capture_factory=FakeCapture,
        )
        self.addCleanup(service.close)
        first = service.start(AimSettings, model_path="first.onnx")
        self.assertTrue(first_path_consumed.wait(1.0))
        duplicate = service.start(AimSettings, model_path="ignored.onnx")
        self.assertEqual(duplicate, first)
        service.stop("switch")
        service.start(AimSettings, model_path="second.onnx")

        self.assertTrue(ready_count.wait(1.0))
        self.assertEqual(paths, ["first.onnx", "second.onnx"])

    def test_detector_and_capture_are_initialized_in_order_off_main_thread(self):
        calls = []
        ready = threading.Event()

        def detector_factory(_path):
            calls.append(("detector", threading.current_thread().name))
            return FakeDetector()

        class ThreadRecordingCapture(FakeCapture):
            def __init__(self):
                calls.append(("capture", threading.current_thread().name))
                super().__init__()

            def start(self):
                calls.append(("start", threading.current_thread().name))

        def sink(event):
            if event.kind == "ready":
                ready.set()

        service = AiService(
            sink,
            detector_factory=detector_factory,
            capture_factory=ThreadRecordingCapture,
        )
        self.addCleanup(service.close)
        generation = service.start(AimSettings)

        self.assertTrue(ready.wait(1.0))
        worker_name = f"AiInference-{generation}"
        self.assertEqual(calls, [
            ("detector", worker_name),
            ("capture", worker_name),
            ("start", worker_name),
        ])

    def test_fps_is_emitted_no_more_than_once_per_second(self):
        events = []
        service = AiService(
            events.append,
            detector_factory=lambda _path: FakeDetector(),
            capture_factory=lambda: FakeCapture([rgb_frame(10), rgb_frame(20)]),
            clock=FakeClock([0.0, 0.1, 0.2, 1.2, 1.3]),
        )
        self.addCleanup(service.close)

        service.start(AimSettings)

        self.assertTrue(wait_until(lambda: any(e.kind == "fps" for e in events)))
        fps_events = [event for event in events if event.kind == "fps"]
        self.assertEqual(len(fps_events), 1)
        self.assertGreater(fps_events[0].payload, 0)

    def test_model_load_error_is_safe_and_stops_worker(self):
        events = []

        def fail_model(_path):
            raise RuntimeError("secret model location")

        service = AiService(
            events.append,
            detector_factory=fail_model,
            capture_factory=FakeCapture,
        )
        self.addCleanup(service.close)

        with self.assertLogs("jitter_app.ai.service", level="ERROR") as logs:
            service.start(AimSettings)
            self.assertTrue(wait_until(lambda: service.status == "error"))
            self.assertTrue(wait_until(lambda: not service.running))

        errors = [event.payload for event in events if event.kind == "error"]
        self.assertEqual(errors, ["RuntimeError: AI service failed"])
        self.assertNotIn("secret model location", errors[0])
        self.assertIn("secret model location", "\n".join(logs.output))
        self.assertIsNone(service.latest_snapshot())
        self.assertIsNone(service.latest_detection_snapshot())

    def test_capture_start_error_closes_capture_exactly_once(self):
        capture = CountingCapture(start_error=OSError("desktop unavailable"))
        events = []
        service = AiService(
            events.append,
            detector_factory=lambda _path: FakeDetector(),
            capture_factory=lambda: capture,
        )
        self.addCleanup(service.close)

        with self.assertLogs("jitter_app.ai.service", level="ERROR"):
            service.start(AimSettings)
            self.assertTrue(capture.closed.wait(1.0))

        self.assertEqual(capture.close_calls, 1)
        self.assertEqual(service.status, "error")
        self.assertEqual(
            [event.payload for event in events if event.kind == "error"],
            ["OSError: AI service failed"],
        )

    def test_malformed_capture_frame_uses_runtime_error_cleanup(self):
        capture = CountingCapture(
            read_error=ValueError(
                "AI capture frame must be nonempty RGB uint8"
            )
        )
        events = []
        service = AiService(
            events.append,
            detector_factory=lambda _path: FakeDetector(),
            capture_factory=lambda: capture,
        )
        self.addCleanup(service.close)

        with self.assertLogs("jitter_app.ai.service", level="ERROR"):
            service.start(AimSettings)
            self.assertTrue(wait_until(
                lambda: any(event.kind == "error" for event in events)
            ))
            self.assertTrue(capture.closed.wait(1.0))

        self.assertEqual(capture.close_calls, 1)
        self.assertIsNone(service.latest_snapshot())
        self.assertIsNone(service.latest_detection_snapshot())
        error = next(event for event in events if event.kind == "error")
        self.assertEqual(error.payload, "ValueError: AI service failed")
        self.assertNotIn("Traceback", error.payload)
        self.assertNotIn("\n", error.payload)

    def test_inference_error_clears_target_and_detection_frame(self):
        capture = CountingCapture([rgb_frame(), rgb_frame()])
        events = []

        class FailingDetector:
            provider = "CPUExecutionProvider"

            def __init__(self):
                self.calls = 0

            def detect(self, _frame):
                self.calls += 1
                if self.calls == 1:
                    return (Detection(150, 150, 170, 170, 0.9, 7),)
                raise ValueError("raw inference diagnostic")

        service = AiService(
            events.append,
            detector_factory=lambda _path: FailingDetector(),
            capture_factory=lambda: capture,
        )
        self.addCleanup(service.close)

        with self.assertLogs("jitter_app.ai.service", level="ERROR"):
            service.start(AimSettings)
            self.assertTrue(capture.closed.wait(1.0))

        self.assertEqual(service.status, "error")
        self.assertIsNone(service.latest_snapshot())
        self.assertIsNone(service.latest_detection_snapshot())
        self.assertEqual(capture.close_calls, 1)
        error = next(event for event in events if event.kind == "error")
        self.assertEqual(error.payload, "ValueError: AI service failed")

    def test_empty_frames_are_skipped_before_inference(self):
        detector = SequenceDetector([
            (Detection(150, 150, 170, 170, 0.9, 7),),
        ])
        service = AiService(
            lambda _event: None,
            detector_factory=lambda _path: detector,
            capture_factory=lambda: FakeCapture([None, None, rgb_frame()]),
        )
        self.addCleanup(service.close)

        service.start(AimSettings)

        self.assertTrue(wait_until(lambda: service.latest_snapshot() is not None))
        self.assertEqual(detector.calls, 1)
        self.assertEqual(service.latest_snapshot().sequence, 1)

    def test_no_target_replaces_previous_snapshot_with_none(self):
        second_entered = threading.Event()
        release_second = threading.Event()

        class NoTargetDetector:
            provider = "CPUExecutionProvider"

            def __init__(self):
                self.calls = 0

            def detect(self, _frame):
                self.calls += 1
                if self.calls == 1:
                    return (Detection(150, 150, 170, 170, 0.9, 7),)
                second_entered.set()
                release_second.wait(1.0)
                return ()

        detector = NoTargetDetector()
        service = AiService(
            lambda _event: None,
            detector_factory=lambda _path: detector,
            capture_factory=lambda: FakeCapture([rgb_frame(), rgb_frame()]),
        )
        self.addCleanup(release_second.set)
        self.addCleanup(service.close)
        service.start(AimSettings)
        self.assertTrue(second_entered.wait(1.0))
        self.assertIsNotNone(service.latest_snapshot())

        release_second.set()

        self.assertTrue(wait_until(lambda: service.latest_snapshot() is None))

    def test_bad_event_sink_does_not_kill_inference(self):
        events = []

        def flaky_sink(event):
            if event.kind == "loading":
                raise RuntimeError("UI queue closed")
            events.append(event)

        service = AiService(
            flaky_sink,
            detector_factory=lambda _path: FakeDetector(),
            capture_factory=FakeCapture,
        )
        self.addCleanup(service.close)

        with self.assertLogs("jitter_app.ai.service", level="ERROR") as logs:
            service.start(AimSettings)
            self.assertTrue(wait_until(lambda: service.latest_snapshot() is not None))

        self.assertEqual(service.status, "ready")
        self.assertIn(AiEvent("ready", "DmlExecutionProvider"), events)
        self.assertIn("UI queue closed", "\n".join(logs.output))

    def test_stop_and_close_are_idempotent_and_cleanup_once(self):
        capture = CountingCapture()
        ready = threading.Event()
        events = []

        def sink(event):
            events.append(event)
            if event.kind == "ready":
                ready.set()

        service = AiService(
            sink,
            detector_factory=lambda _path: FakeDetector(),
            capture_factory=lambda: capture,
        )
        service.start(AimSettings)
        self.assertTrue(ready.wait(1.0))

        service.stop("manual")
        service.stop("manual")
        service.close()
        service.close()

        self.assertTrue(capture.closed.wait(1.0))
        self.assertEqual(capture.close_calls, 1)
        self.assertEqual(
            [event for event in events if event.kind == "stopped"],
            [AiEvent("stopped", "manual")],
        )
        self.assertIsNone(service.start(AimSettings))

    def test_close_invalidates_worker_without_post_close_events(self):
        detector = BlockingDetector()
        events = []
        capture = CountingCapture()
        service = AiService(
            events.append,
            detector_factory=lambda _path: detector,
            capture_factory=lambda: capture,
        )
        service.start(AimSettings)
        self.assertTrue(detector.entered.wait(1.0))
        event_count = len(events)

        service.close()
        detector.release.set()

        self.assertTrue(capture.closed.wait(1.0))
        self.assertEqual(len(events), event_count)
        self.assertFalse(service.running)
        self.assertEqual(service.status, "stopped")
        self.assertIsNone(service.latest_snapshot())
        self.assertIsNone(service.latest_detection_snapshot())

    def test_stop_orders_stopped_after_an_inflight_worker_event(self):
        fps_entered = threading.Event()
        fps_delivered = threading.Event()
        release_fps = threading.Event()
        stop_returns = [threading.Event(), threading.Event()]
        events = []

        def sink(event):
            if event.kind == "fps":
                fps_entered.set()
                release_fps.wait(1.0)
            events.append(event)
            if event.kind == "fps":
                fps_delivered.set()

        service = AiService(
            sink,
            detector_factory=lambda _path: FakeDetector(),
            capture_factory=lambda: FakeCapture([rgb_frame(10), rgb_frame(20)]),
            clock=FakeClock([0.0, 0.1, 0.2, 1.2, 1.3]),
        )
        self.addCleanup(release_fps.set)
        self.addCleanup(service.close)
        service.start(AimSettings)
        self.assertTrue(fps_entered.wait(1.0))

        def stop_service(returned):
            service.stop("manual")
            returned.set()

        stoppers = [
            threading.Thread(target=stop_service, args=(returned,), daemon=True)
            for returned in stop_returns
        ]
        for stopper in stoppers:
            stopper.start()
        self.assertTrue(wait_until(lambda: not service.running))
        release_fps.set()
        self.assertTrue(all(returned.wait(1.0) for returned in stop_returns))
        self.assertTrue(fps_delivered.wait(1.0))

        self.assertEqual(events[-2:], [
            AiEvent("fps", 2 / 1.3),
            AiEvent("stopped", "manual"),
        ])

    def test_all_concurrent_stop_and_close_callers_cross_event_barrier(self):
        fps_entered = threading.Event()
        fps_delivered = threading.Event()
        release_fps = threading.Event()
        events = []

        def sink(event):
            if event.kind == "fps":
                fps_entered.set()
                release_fps.wait(1.0)
            events.append(event)
            if event.kind == "fps":
                fps_delivered.set()

        service = AiService(
            sink,
            detector_factory=lambda _path: FakeDetector(),
            capture_factory=lambda: FakeCapture([rgb_frame(10), rgb_frame(20)]),
            clock=FakeClock([0.0, 0.1, 0.2, 1.2, 1.3]),
        )
        observed_lock = ObservedRLock()
        service._event_lock = observed_lock
        self.addCleanup(release_fps.set)
        self.addCleanup(service.close)
        service.start(AimSettings)
        self.assertTrue(fps_entered.wait(1.0))
        observed_lock.reset_attempts()

        calls = [
            lambda: service.stop("owner"),
            lambda: service.stop("duplicate"),
            service.close,
            service.close,
        ]
        entered = [threading.Event() for _call in calls]
        returned = [threading.Event() for _call in calls]

        def invoke(call, entry, result):
            entry.set()
            call()
            result.set()

        owner = threading.Thread(
            target=invoke,
            args=(calls[0], entered[0], returned[0]),
            daemon=True,
        )
        owner.start()
        self.assertTrue(wait_until(lambda: not service.running))

        others = [
            threading.Thread(
                target=invoke,
                args=(calls[index], entered[index], returned[index]),
                daemon=True,
            )
            for index in range(1, len(calls))
        ]
        for caller in others:
            caller.start()
        self.assertTrue(all(event.wait(1.0) for event in entered))
        self.assertTrue(wait_until(
            lambda: observed_lock.attempts == len(calls)
            or any(event.is_set() for event in returned)
        ))
        self.assertFalse(any(event.is_set() for event in returned))

        release_fps.set()

        self.assertTrue(all(event.wait(1.0) for event in returned))
        self.assertTrue(fps_delivered.wait(1.0))
        self.assertEqual(
            len([event for event in events if event.kind == "stopped"]),
            0,
        )

    def test_old_generation_cannot_overwrite_new_snapshot(self):
        old_detector = BlockingDetector()
        new_detector = FakeDetector()
        detectors = iter([old_detector, new_detector])
        old_capture = CountingCapture([rgb_frame(150)])
        new_capture = CountingCapture([rgb_frame(30)])
        captures = iter([old_capture, new_capture])
        service = AiService(
            lambda _event: None,
            detector_factory=lambda _path: next(detectors),
            capture_factory=lambda: next(captures),
        )
        self.addCleanup(old_detector.release.set)
        self.addCleanup(service.close)
        service.start(AimSettings)
        self.assertTrue(old_detector.entered.wait(1.0))

        service.stop("restart")
        service.start(AimSettings)
        self.assertTrue(wait_until(
            lambda: service.latest_snapshot() is not None
            and service.latest_snapshot().aim_x == 35.0
        ))

        old_detector.release.set()

        self.assertTrue(old_capture.closed.wait(1.0))
        self.assertEqual(service.latest_snapshot().aim_x, 35.0)
        self.assertEqual(
            service.latest_detection_snapshot().detections,
            (Detection(30, 30, 40, 40, 0.9, 7),),
        )
        self.assertEqual(old_capture.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
