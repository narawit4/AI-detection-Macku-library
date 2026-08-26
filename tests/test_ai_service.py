import threading
import time
import unittest

from ai_service import AiEvent, AiService
from ai_targeting import AimSettings, Detection


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    wake = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return True
        wake.wait(0.005)
    return predicate()


class FakeCapture:
    def __init__(self, frames=None):
        self.frames = list(frames or [10])
        self.closed = threading.Event()

    def start(self):
        pass

    def read(self):
        if self.frames:
            return self.frames.pop(0)
        return None

    def close(self):
        self.closed.set()


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

    def detect(self, frame):
        return (
            Detection(frame, frame, frame + 10, frame + 10, 0.9, 7),
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


class AiServiceTests(unittest.TestCase):
    def test_ready_worker_publishes_only_latest_snapshot(self):
        events = []
        service = AiService(
            events.append,
            detector_factory=lambda path: FakeDetector(),
            capture_factory=lambda: FakeCapture(frames=[10, 20]),
            clock=FakeClock([1.0, 1.01, 1.02, 2.1]),
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
            capture_factory=lambda: FakeCapture([10, 20]),
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

        with self.assertLogs("ai_service", level="ERROR") as logs:
            service.start(AimSettings)
            self.assertTrue(wait_until(lambda: service.status == "error"))
            self.assertTrue(wait_until(lambda: not service.running))

        errors = [event.payload for event in events if event.kind == "error"]
        self.assertEqual(errors, ["RuntimeError: AI service failed"])
        self.assertNotIn("secret model location", errors[0])
        self.assertIn("secret model location", "\n".join(logs.output))
        self.assertIsNone(service.latest_snapshot())

    def test_capture_start_error_closes_capture_exactly_once(self):
        capture = CountingCapture(start_error=OSError("desktop unavailable"))
        events = []
        service = AiService(
            events.append,
            detector_factory=lambda _path: FakeDetector(),
            capture_factory=lambda: capture,
        )
        self.addCleanup(service.close)

        with self.assertLogs("ai_service", level="ERROR"):
            service.start(AimSettings)
            self.assertTrue(capture.closed.wait(1.0))

        self.assertEqual(capture.close_calls, 1)
        self.assertEqual(service.status, "error")
        self.assertEqual(
            [event.payload for event in events if event.kind == "error"],
            ["OSError: AI service failed"],
        )

    def test_inference_error_clears_target_and_closes_capture(self):
        capture = CountingCapture([object()])
        events = []

        class FailingDetector:
            provider = "CPUExecutionProvider"

            def detect(self, _frame):
                raise ValueError("raw inference diagnostic")

        service = AiService(
            events.append,
            detector_factory=lambda _path: FailingDetector(),
            capture_factory=lambda: capture,
        )
        self.addCleanup(service.close)

        with self.assertLogs("ai_service", level="ERROR"):
            service.start(AimSettings)
            self.assertTrue(capture.closed.wait(1.0))

        self.assertEqual(service.status, "error")
        self.assertIsNone(service.latest_snapshot())
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
            capture_factory=lambda: FakeCapture([None, None, object()]),
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
            capture_factory=lambda: FakeCapture([object(), object()]),
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

        with self.assertLogs("ai_service", level="ERROR") as logs:
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
            capture_factory=lambda: FakeCapture([10, 20]),
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

    def test_old_generation_cannot_overwrite_new_snapshot(self):
        old_detector = BlockingDetector()
        new_detector = FakeDetector()
        detectors = iter([old_detector, new_detector])
        old_capture = CountingCapture([150])
        new_capture = CountingCapture([30])
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
        self.assertEqual(old_capture.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
