import threading
import time
import unittest
from unittest import mock

from makcu import MouseButton

from ai_targeting import AimSettings, TargetSnapshot
from makcu_service import BUTTON_NAMES, MakcuService, ServiceEvent
from motion import MotionSettings


class FakeController:
    def __init__(self):
        self.connection_callback = None
        self.button_callback = None
        self.monitoring = False
        self.disconnected = False
        self.moves = []

    def on_connection_change(self, callback):
        self.connection_callback = callback

    def enable_button_monitoring(self, enabled):
        self.monitoring = enabled

    def set_button_callback(self, callback):
        self.button_callback = callback

    def get_device_info(self):
        return "Fake Makcu"

    def get_firmware_version(self):
        return "1.0"

    def move(self, x, y):
        self.moves.append((x, y))

    def disconnect(self):
        self.disconnected = True


class RecordingEngine:
    def __init__(self):
        self.calls = []

    def step(self, _settings, _dt, _elapsed):
        self.calls.append((_settings, _dt, _elapsed))
        return 0, -1


class FakeAimEngine:
    def __init__(self):
        self.calls = []
        self.polled_again = threading.Event()

    def step(self, snapshot, settings, now):
        self.calls.append((snapshot, settings, now))
        if len(self.calls) > 1:
            self.polled_again.set()
            return 0, 0
        return 3, 0


class BlockingAimEngine:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def step(self, _snapshot, _settings, _now):
        self.entered.set()
        if not self.release.wait(1.0):
            raise TimeoutError("test did not release blocked AI engine")
        return 3, 0


class StopAfterFirstWait:
    def __init__(self):
        self.timeouts = []
        self._set = False

    def is_set(self):
        return self._set

    def wait(self, timeout):
        self.timeouts.append(timeout)
        self._set = True
        return True


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class MakcuConnectionTests(unittest.TestCase):
    def test_button_names_cover_the_supported_buttons(self):
        self.assertEqual(
            [BUTTON_NAMES[button] for button in MouseButton],
            ["Left", "Right", "Middle", "Mouse4", "Mouse5"],
        )

    def test_successful_worker_configures_controller_and_emits_connected(self):
        controller = FakeController()
        events = []
        service = MakcuService(events.append, controller_factory=lambda **_kwargs: controller)
        generation = service._begin_connection()
        service._connect_worker(generation)
        self.assertTrue(controller.monitoring)
        self.assertIsNotNone(controller.connection_callback)
        self.assertIsNotNone(controller.button_callback)
        self.assertTrue(service.connected)
        self.assertEqual(events[-1].kind, "connected")
        self.assertIn("Fake Makcu", events[-1].payload)

    def test_connection_failure_emits_disconnected_without_controller(self):
        events = []

        def failing_factory(**_kwargs):
            raise RuntimeError("not found")

        service = MakcuService(events.append, controller_factory=failing_factory)
        generation = service._begin_connection()
        service._connect_worker(generation)
        self.assertFalse(service.connected)
        self.assertEqual(events[-1], ServiceEvent("disconnected", "RuntimeError: not found"))

    def test_old_generation_controller_is_disconnected_and_ignored(self):
        old = FakeController()
        service = MakcuService(lambda _event: None, controller_factory=lambda **_kwargs: old)
        generation = service._begin_connection()
        service._begin_connection()
        service._connect_worker(generation)
        self.assertTrue(old.disconnected)
        self.assertIsNone(service.controller)

    def test_disconnect_signal_during_setup_prevents_controller_install(self):
        class DropsDuringSetup(FakeController):
            def on_connection_change(self, callback):
                super().on_connection_change(callback)
                callback(False)

        controller = DropsDuringSetup()
        events = []
        service = MakcuService(events.append, controller_factory=lambda **_kwargs: controller)
        generation = service._begin_connection()
        service._connect_worker(generation)
        self.assertTrue(controller.disconnected)
        self.assertFalse(service.connected)
        self.assertIsNone(service.controller)
        self.assertEqual(events[-1].kind, "disconnected")

    def test_button_callback_emits_normalized_name_and_pressed_state(self):
        controller = FakeController()
        events = []
        service = MakcuService(events.append, controller_factory=lambda **_kwargs: controller)
        generation = service._begin_connection()
        service._connect_worker(generation)
        service._button_event(MouseButton.LEFT, True)
        self.assertEqual(events[-1], ServiceEvent("button", ("Left", True)))
        service._button_event("Mouse5", False)
        self.assertEqual(events[-1], ServiceEvent("button", ("Mouse5", False)))

    def test_stale_button_callback_is_ignored_after_reconnect(self):
        old = FakeController()
        fresh = FakeController()
        controllers = iter((old, fresh))
        events = []
        service = MakcuService(events.append, controller_factory=lambda **_kwargs: next(controllers))
        first_generation = service._begin_connection()
        service._connect_worker(first_generation)
        old_button_callback = old.button_callback
        service.reconnect()
        self.assertTrue(wait_until(lambda: service.controller is fresh))
        events_before_stale_callback = list(events)
        old_button_callback(MouseButton.LEFT, True)
        self.assertEqual(events, events_before_stale_callback)
        fresh.button_callback(MouseButton.RIGHT, True)
        self.assertEqual(events[-1], ServiceEvent("button", ("Right", True)))


class MakcuMovementTests(unittest.TestCase):
    def connected_service(
        self, *, controller=None, use_default_engine=False, **service_kwargs
    ):
        controller = controller or FakeController()
        events = []
        kwargs = {}
        if not use_default_engine:
            kwargs["engine_factory"] = RecordingEngine
        kwargs.update(service_kwargs)
        service = MakcuService(
            events.append, controller_factory=lambda **_kwargs: controller, **kwargs
        )
        generation = service._begin_connection()
        service._connect_worker(generation)
        self.addCleanup(service.close)
        return service, controller, events

    def test_ai_motion_uses_controller_and_consumes_latest_snapshot_once(self):
        engine = FakeAimEngine()
        service, controller, _events = self.connected_service(
            aim_engine_factory=lambda: engine
        )
        snapshot = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)

        self.assertTrue(service.start_ai_motion(lambda: snapshot, AimSettings))
        self.assertTrue(engine.polled_again.wait(1.0))
        service.stop_motion("manual")
        service.join_motion(1.0)

        self.assertEqual(controller.moves, [(3, 0)])
        self.assertIs(engine.calls[0][0], snapshot)
        self.assertEqual(engine.calls[0][1], AimSettings())

    def test_ai_stop_return_barrier_prevents_late_move(self):
        engine = BlockingAimEngine()
        target = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)
        service, controller, _events = self.connected_service(
            aim_engine_factory=lambda: engine
        )
        self.addCleanup(engine.release.set)
        self.assertTrue(service.start_ai_motion(lambda: target, AimSettings))
        self.assertTrue(engine.entered.wait(1.0))

        service.stop_motion("manual")
        engine.release.set()
        service.join_motion(1.0)

        self.assertEqual(controller.moves, [])

    def test_ai_motion_rejects_disconnected_service_and_invalid_duration(self):
        disconnected = MakcuService(lambda _event: None)
        self.addCleanup(disconnected.close)
        self.assertFalse(
            disconnected.start_ai_motion(lambda: None, AimSettings)
        )

        service, _controller, _events = self.connected_service()
        for invalid_duration in (object(), "not-a-duration"):
            with self.subTest(duration=invalid_duration):
                self.assertFalse(
                    service.start_ai_motion(
                        lambda: None, AimSettings, invalid_duration
                    )
                )

    def test_ai_motion_is_mutually_exclusive_with_jitter_motion(self):
        ai_engine = BlockingAimEngine()
        jitter_engines = []
        service, _controller, _events = self.connected_service(
            engine_factory=lambda: jitter_engines.append(RecordingEngine()),
            aim_engine_factory=lambda: ai_engine,
        )
        target = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)
        self.addCleanup(ai_engine.release.set)

        self.assertTrue(service.start_ai_motion(lambda: target, AimSettings))
        self.assertTrue(ai_engine.entered.wait(1.0))
        active_thread = service._motion_thread
        self.assertTrue(service.start_motion(lambda: MotionSettings()))
        self.assertTrue(service.start_ai_motion(lambda: target, AimSettings))

        self.assertIs(service._motion_thread, active_thread)
        self.assertEqual(jitter_engines, [])
        service.stop_motion("manual")
        ai_engine.release.set()
        service.join_motion(1.0)

    def test_ai_duration_completion_emits_motion_stopped(self):
        service, controller, events = self.connected_service()

        self.assertTrue(
            service.start_ai_motion(lambda: None, AimSettings, duration_s=0)
        )
        service.join_motion(1.0)

        self.assertFalse(service.motion_active)
        self.assertEqual(controller.moves, [])
        self.assertEqual(
            events[-1], ServiceEvent("motion_stopped", "duration_complete")
        )

    def test_ai_snapshot_and_settings_provider_exceptions_emit_motion_error(self):
        def snapshot_failure():
            raise ValueError("snapshot failed")

        service, _controller, events = self.connected_service()
        self.assertTrue(service.start_ai_motion(snapshot_failure, AimSettings))
        service.join_motion(1.0)
        self.assertEqual(
            events[-1], ServiceEvent("motion_error", "ValueError: snapshot failed")
        )

        def settings_failure():
            raise LookupError("settings failed")

        service, _controller, events = self.connected_service()
        self.assertTrue(service.start_ai_motion(lambda: None, settings_failure))
        service.join_motion(1.0)
        self.assertEqual(
            events[-1], ServiceEvent("motion_error", "LookupError: settings failed")
        )

    def test_ai_disconnect_cancels_before_move_and_emits_disconnected(self):
        engine = BlockingAimEngine()
        target = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)
        service, controller, events = self.connected_service(
            aim_engine_factory=lambda: engine
        )
        self.addCleanup(engine.release.set)
        self.assertTrue(service.start_ai_motion(lambda: target, AimSettings))
        self.assertTrue(engine.entered.wait(1.0))

        disconnect_returned = threading.Event()

        def disconnect():
            service._connection_changed(service.connection_generation, False)
            disconnect_returned.set()

        thread = threading.Thread(target=disconnect)
        thread.start()
        try:
            self.assertTrue(service._motion_stop.wait(1.0))
            engine.release.set()
            self.assertTrue(disconnect_returned.wait(1.0))
            service.join_motion(1.0)
        finally:
            engine.release.set()
            thread.join(1.0)

        self.assertEqual(controller.moves, [])
        self.assertFalse(service.connected)
        self.assertEqual(events[-1], ServiceEvent("disconnected"))

    def test_ai_controller_exception_emits_motion_error(self):
        class FailingController(FakeController):
            def move(self, _x, _y):
                raise RuntimeError("AI move failed")

        engine = FakeAimEngine()
        service, _controller, events = self.connected_service(
            controller=FailingController(), aim_engine_factory=lambda: engine
        )
        snapshot = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)

        self.assertTrue(service.start_ai_motion(lambda: snapshot, AimSettings))
        service.join_motion(1.0)

        self.assertFalse(service.motion_active)
        self.assertEqual(
            events[-1], ServiceEvent("motion_error", "RuntimeError: AI move failed")
        )

    def test_ai_manual_stop_emits_motion_stopped_reason(self):
        engine = BlockingAimEngine()
        target = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)
        service, controller, events = self.connected_service(
            aim_engine_factory=lambda: engine
        )
        self.addCleanup(engine.release.set)
        self.assertTrue(service.start_ai_motion(lambda: target, AimSettings))
        self.assertTrue(engine.entered.wait(1.0))

        service.stop_motion("ai_disabled")
        engine.release.set()
        service.join_motion(1.0)

        self.assertEqual(controller.moves, [])
        self.assertEqual(
            events[-1], ServiceEvent("motion_stopped", "ai_disabled")
        )

    def test_close_during_ai_motion_cancels_before_disconnect(self):
        engine = BlockingAimEngine()
        target = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)
        service, controller, _events = self.connected_service(
            aim_engine_factory=lambda: engine
        )
        self.addCleanup(engine.release.set)
        self.assertTrue(service.start_ai_motion(lambda: target, AimSettings))
        self.assertTrue(engine.entered.wait(1.0))

        service.close()
        engine.release.set()
        service.join_motion(1.0)

        self.assertEqual(controller.moves, [])
        self.assertFalse(service.connected)
        self.assertTrue(wait_until(lambda: controller.disconnected))

    def test_ai_worker_polls_at_240_hz(self):
        engine = FakeAimEngine()
        service, _controller, _events = self.connected_service(
            aim_engine_factory=lambda: engine
        )
        stop_event = StopAfterFirstWait()
        connection_generation = service.connection_generation
        with service._lock:
            service._motion_generation += 1
            motion_generation = service._motion_generation
            service._motion_stop_reasons[motion_generation] = None
            service._motion_active = True
            service._motion_thread = threading.current_thread()
        snapshot = TargetSnapshot(1, 100.0, "head", 170, 160)

        with mock.patch("makcu_service.time.perf_counter", return_value=100.0):
            service._motion_worker(
                motion_generation,
                connection_generation,
                stop_event,
                AimSettings,
                None,
                "ai",
                lambda: snapshot,
            )

        self.assertEqual(stop_event.timeouts, [1 / 240])

    def test_paired_pulse_worker_sends_vertical_reports_and_stop_prevents_next_half(self):
        service, controller, _events = self.connected_service(use_default_engine=True)
        self.assertTrue(
            service.start_motion(lambda: MotionSettings(2, 20, "Instant"))
        )
        self.assertTrue(wait_until(lambda: len(controller.moves) >= 1))
        service.stop_motion("manual")
        moves_after_stop = list(controller.moves)
        time.sleep(0.06)
        self.assertEqual(controller.moves, moves_after_stop)
        self.assertTrue(
            all(x == 0 and abs(y) <= 2 for x, y in moves_after_stop)
        )

    def test_worker_waits_one_half_pulse_interval(self):
        service, _controller, _events = self.connected_service()
        stop_event = StopAfterFirstWait()
        connection_generation = service.connection_generation
        with service._lock:
            service._motion_generation += 1
            motion_generation = service._motion_generation
            service._motion_stop_reasons[motion_generation] = None
            service._motion_active = True
            service._motion_thread = threading.current_thread()
        with mock.patch("makcu_service.time.perf_counter", return_value=100.0):
            service._motion_worker(
                motion_generation,
                connection_generation,
                stop_event,
                lambda: MotionSettings(2, 20, "Instant"),
                None,
            )
        self.assertEqual(stop_event.timeouts, [0.025])

    def test_stop_signals_while_move_is_blocked_and_serializes_its_return(self):
        class GatedController(FakeController):
            def __init__(self):
                super().__init__()
                self.move_entered = threading.Event()
                self.release_move = threading.Event()
                self.move_starts = 0

            def move(self, x, y):
                self.move_starts += 1
                self.move_entered.set()
                if not self.release_move.wait(1.0):
                    raise TimeoutError("test did not release blocked move")
                super().move(x, y)

        controller = GatedController()
        events = []
        service = MakcuService(
            events.append,
            controller_factory=lambda **_kwargs: controller,
            engine_factory=RecordingEngine,
        )
        generation = service._begin_connection()
        service._connect_worker(generation)
        self.assertTrue(service.start_motion(lambda: MotionSettings()))
        self.assertTrue(controller.move_entered.wait(1.0))

        stop_started = threading.Event()
        stop_returned = threading.Event()

        def request_stop():
            stop_started.set()
            service.stop_motion("gated_stop")
            stop_returned.set()

        stop_thread = threading.Thread(target=request_stop)
        stop_thread.start()
        try:
            self.assertTrue(stop_started.wait(1.0))
            self.assertTrue(service._motion_stop.wait(0.2))
            self.assertFalse(stop_returned.is_set())
            controller.release_move.set()
            self.assertTrue(stop_returned.wait(1.0))
            move_starts_when_stop_returned = controller.move_starts
            service.join_motion(1.0)
            self.assertFalse(service.motion_active)
            self.assertEqual(controller.move_starts, move_starts_when_stop_returned)
            self.assertEqual(controller.move_starts, 1)
            self.assertEqual(
                events[-1], ServiceEvent("motion_stopped", "gated_stop")
            )
        finally:
            controller.release_move.set()
            stop_thread.join(1.0)
            service.join_motion(1.0)

    def test_timed_motion_finishes_and_emits_test_complete(self):
        service, controller, events = self.connected_service()
        self.assertTrue(
            service.start_motion(
                lambda: MotionSettings(2, pulse_rate_hz=20, ramp_mode="Instant"),
                duration_s=0.02,
            )
        )
        service.join_motion(1.0)
        self.assertGreater(len(controller.moves), 0)
        self.assertEqual(events[-1], ServiceEvent("motion_stopped", "duration_complete"))

    def test_start_motion_is_idempotent_and_rejects_disconnected_service(self):
        service, _controller, _events = self.connected_service()
        self.assertTrue(service.start_motion(lambda: MotionSettings()))
        self.assertTrue(service.start_motion(lambda: MotionSettings()))
        service.stop_motion()
        service.join_motion(1.0)
        service._connection_changed(service.connection_generation, False)
        self.assertFalse(service.start_motion(lambda: MotionSettings()))

    def test_move_exception_is_contained_and_emits_motion_error(self):
        class FailingController(FakeController):
            def move(self, _x, _y):
                raise RuntimeError("move failed")

        controller = FailingController()
        events = []
        service = MakcuService(
            events.append,
            controller_factory=lambda **_kwargs: controller,
            engine_factory=RecordingEngine,
        )
        generation = service._begin_connection()
        service._connect_worker(generation)
        self.assertTrue(service.start_motion(lambda: MotionSettings()))
        service.join_motion(1.0)
        self.assertFalse(service.motion_active)
        self.assertEqual(events[-1], ServiceEvent("motion_error", "RuntimeError: move failed"))

    def test_disconnect_stops_motion_before_emitting_disconnected(self):
        service, _controller, events = self.connected_service()
        service.start_motion(lambda: MotionSettings())
        generation = service.connection_generation
        service._connection_changed(generation, False)
        service.join_motion(1.0)
        self.assertFalse(service.motion_active)
        self.assertEqual(events[-1].kind, "disconnected")


class MakcuConnectionLifecycleTests(unittest.TestCase):
    def test_connect_starts_a_daemon_worker(self):
        worker_seen = threading.Event()
        daemon_state = []
        controller = FakeController()

        def factory(**_kwargs):
            daemon_state.append(threading.current_thread().daemon)
            worker_seen.set()
            return controller

        service = MakcuService(lambda _event: None, controller_factory=factory)
        service.connect()
        self.assertTrue(worker_seen.wait(1.0))
        self.assertEqual(daemon_state, [True])

    def test_reconnect_disconnects_exact_old_controller(self):
        old = FakeController()
        fresh = FakeController()
        controllers = iter((old, fresh))
        service = MakcuService(lambda _event: None, controller_factory=lambda **_kwargs: next(controllers))
        first_generation = service._begin_connection()
        service._connect_worker(first_generation)
        service.reconnect()
        self.assertTrue(wait_until(lambda: old.disconnected))
        self.assertIs(service.controller, fresh)

    def test_stale_connection_callback_cannot_change_current_state(self):
        old = FakeController()
        fresh = FakeController()
        controllers = iter((old, fresh))
        events = []
        service = MakcuService(events.append, controller_factory=lambda **_kwargs: next(controllers))
        first_generation = service._begin_connection()
        service._connect_worker(first_generation)
        service.reconnect()
        self.assertTrue(wait_until(lambda: service.controller is fresh))
        old.connection_callback(False)
        self.assertTrue(service.connected)
        self.assertIs(service.controller, fresh)

    def test_diagnostic_failure_does_not_fail_connection(self):
        class NoDiagnostics(FakeController):
            def get_device_info(self):
                raise RuntimeError("diagnostics unavailable")

            def get_firmware_version(self):
                raise RuntimeError("diagnostics unavailable")

        controller = NoDiagnostics()
        events = []
        service = MakcuService(events.append, controller_factory=lambda **_kwargs: controller)
        generation = service._begin_connection()
        service._connect_worker(generation)
        self.assertTrue(service.connected)
        self.assertEqual(events[-1].kind, "connected")

    def test_connected_event_cannot_follow_a_disconnect_race(self):
        controller = FakeController()
        entered_connected_sink = threading.Event()
        release_connected_sink = threading.Event()
        events = []
        service = None

        def sink(event):
            if event.kind == "connected":
                entered_connected_sink.set()
                self.assertTrue(release_connected_sink.wait(1.0))
            events.append(event)

        service = MakcuService(sink, controller_factory=lambda **_kwargs: controller)
        generation = service._begin_connection()
        worker = threading.Thread(target=service._connect_worker, args=(generation,))
        worker.start()
        self.assertTrue(entered_connected_sink.wait(1.0))
        disconnect_thread = threading.Thread(target=lambda: controller.connection_callback(False))
        disconnect_thread.start()
        release_connected_sink.set()
        worker.join(1.0)
        disconnect_thread.join(1.0)
        self.assertFalse(service.connected)
        self.assertEqual(events[-1], ServiceEvent("disconnected"))

    def test_close_invalidates_generation_and_disconnects_controller(self):
        controller = FakeController()
        service = MakcuService(lambda _event: None, controller_factory=lambda **_kwargs: controller)
        generation = service._begin_connection()
        service._connect_worker(generation)
        service.close()
        self.assertTrue(wait_until(lambda: controller.disconnected))
        self.assertFalse(service.connected)
        self.assertIsNone(service.controller)

    def test_connect_replaces_and_disconnects_existing_controller(self):
        old = FakeController()
        fresh = FakeController()
        controllers = iter((old, fresh))
        service = MakcuService(lambda _event: None, controller_factory=lambda **_kwargs: next(controllers))
        first_generation = service._begin_connection()
        service._connect_worker(first_generation)
        service.connect()
        self.assertTrue(wait_until(lambda: old.disconnected))
        self.assertTrue(wait_until(lambda: service.controller is fresh))


if __name__ == "__main__":
    unittest.main()
