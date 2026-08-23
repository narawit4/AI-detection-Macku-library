import threading
import time
import unittest

from makcu import MouseButton

from makcu_service import BUTTON_NAMES, MakcuService, ServiceEvent


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

    def test_close_invalidates_generation_and_disconnects_controller(self):
        controller = FakeController()
        service = MakcuService(lambda _event: None, controller_factory=lambda **_kwargs: controller)
        generation = service._begin_connection()
        service._connect_worker(generation)
        service.close()
        self.assertTrue(wait_until(lambda: controller.disconnected))
        self.assertFalse(service.connected)
        self.assertIsNone(service.controller)


if __name__ == "__main__":
    unittest.main()
