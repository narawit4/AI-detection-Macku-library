import threading
import time
import unittest

from makcu import MouseButton

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


class ConstantEngine:
    def step(self, _settings, _dt, _elapsed):
        return 2, -1


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
    def connected_service(self):
        controller = FakeController()
        events = []
        service = MakcuService(
            events.append,
            controller_factory=lambda **_kwargs: controller,
            engine_factory=ConstantEngine,
        )
        generation = service._begin_connection()
        service._connect_worker(generation)
        return service, controller, events

    def test_start_motion_sends_reports_and_stop_is_interruptible(self):
        service, controller, _events = self.connected_service()
        self.assertTrue(service.start_motion(lambda: MotionSettings()))
        deadline = threading.Event()
        for _index in range(100):
            if controller.moves:
                break
            deadline.wait(0.005)
        service.stop_motion()
        service.join_motion(1.0)
        count_after_stop = len(controller.moves)
        deadline.wait(0.03)
        self.assertGreater(count_after_stop, 0)
        self.assertEqual(len(controller.moves), count_after_stop)

    def test_timed_motion_finishes_and_emits_test_complete(self):
        service, controller, events = self.connected_service()
        self.assertTrue(
            service.start_motion(
                lambda: MotionSettings(update_rate_hz=500), duration_s=0.02
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
            engine_factory=ConstantEngine,
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
