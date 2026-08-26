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
        self.connected = True
        self.health_polls = 0

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

    def is_connected(self):
        self.health_polls += 1
        return self.connected


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


class BlockingFailingAimEngine:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def step(self, _snapshot, _settings, _now):
        self.entered.set()
        if not self.release.wait(1.0):
            raise TimeoutError("test did not release failing AI engine")
        raise RuntimeError("obsolete AI failure")


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
    def test_health_failure_logs_once_per_streak_without_exposing_detail(self):
        private_detail = "token=HEALTH-SECRET at C:\\private\\health.py:71"

        class HealthFailureController(FakeController):
            def is_connected(self):
                self.health_polls += 1
                raise RuntimeError(private_detail)

        controller = HealthFailureController()
        events = []
        service = MakcuService(
            events.append,
            controller_factory=lambda **_kwargs: controller,
        )
        self.addCleanup(service.close)

        with self.assertLogs("makcu_service", level="ERROR") as captured:
            service._connect_worker(service._begin_connection())
            self.assertTrue(wait_until(
                lambda: any(event.kind == "disconnected" for event in events)
            ))
            self.assertTrue(wait_until(lambda: controller.health_polls >= 3))

        health_records = [
            record
            for record in captured.records
            if record.getMessage() == "Makcu connection health check failed"
        ]
        self.assertEqual(len(health_records), 1)
        self.assertIsNotNone(health_records[0].exc_info)
        self.assertIn(private_detail, "\n".join(captured.output))
        self.assertFalse(any(
            "HEALTH-SECRET" in str(event.payload)
            or "C:\\private" in str(event.payload)
            for event in events
        ))

    def test_setup_disconnect_then_failure_emits_only_one_safe_loss_event(self):
        private_detail = "token=SETUP-SECRET at C:\\private\\setup.py:123"

        class DisconnectingSetupFailure(FakeController):
            def on_connection_change(self, callback):
                super().on_connection_change(callback)
                callback(False)

            def enable_button_monitoring(self, _enabled):
                raise RuntimeError(private_detail)

        controller = DisconnectingSetupFailure()
        events = []
        service = MakcuService(
            events.append,
            controller_factory=lambda **_kwargs: controller,
        )

        with self.assertLogs("makcu_service", level="ERROR"):
            service._connect_worker(service._begin_connection())

        loss_events = [event for event in events if event.kind == "disconnected"]
        self.assertEqual(len(loss_events), 1)
        self.assertFalse(any(
            "SETUP-SECRET" in str(event.payload)
            or "C:\\private" in str(event.payload)
            for event in loss_events
        ))

    def test_health_monitor_reports_each_silent_transport_transition_once(self):
        events = []
        controller = FakeController()
        service = MakcuService(
            events.append,
            controller_factory=lambda **_kwargs: controller,
        )
        self.addCleanup(service.close)

        service._connect_worker(service._begin_connection())
        health_thread = service._health_thread
        self.assertIsNotNone(health_thread)

        controller.connected = False
        self.assertTrue(wait_until(
            lambda: [event.kind for event in events].count("disconnected") == 1
        ))
        controller.connected = True
        self.assertTrue(wait_until(
            lambda: [event.kind for event in events].count("reconnected") == 1
        ))
        controller.connected = False
        self.assertTrue(wait_until(
            lambda: [event.kind for event in events].count("disconnected") == 2
        ))
        polls_after_second_loss = controller.health_polls
        self.assertTrue(wait_until(
            lambda: controller.health_polls >= polls_after_second_loss + 2
        ))

        self.assertEqual(
            [event.kind for event in events],
            [
                "connecting",
                "connected",
                "disconnected",
                "reconnected",
                "disconnected",
            ],
        )
        self.assertIs(service._health_thread, health_thread)
        self.assertTrue(health_thread.is_alive())

    def test_physical_transport_loss_emits_disconnected(self):
        events = []
        controller = FakeController()
        service = MakcuService(events.append, controller_factory=lambda **_kwargs: controller)
        self.addCleanup(service.close)

        service._connect_worker(service._begin_connection())
        controller.connected = False

        self.assertTrue(wait_until(
            lambda: any(event.kind == "disconnected" for event in events)
        ))
        self.assertFalse(service.connected)

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

    def test_factory_failure_logs_traceback_but_emits_fixed_safe_message(self):
        events = []
        private_detail = "api_key=TOP-SECRET at C:\\private\\factory.py:417"

        def failing_factory(**_kwargs):
            raise RuntimeError(private_detail)

        service = MakcuService(events.append, controller_factory=failing_factory)
        generation = service._begin_connection()
        with self.assertLogs("makcu_service", level="ERROR") as captured:
            service._connect_worker(generation)

        self.assertFalse(service.connected)
        self.assertEqual(
            events[-1],
            ServiceEvent(
                "disconnected",
                "Makcu unavailable; check USB and reconnect",
            ),
        )
        self.assertNotIn("TOP-SECRET", str(events[-1].payload))
        self.assertNotIn("C:\\private", str(events[-1].payload))
        self.assertIn(private_detail, "\n".join(captured.output))
        self.assertTrue(any(record.exc_info is not None for record in captured.records))

    def test_setup_failure_logs_traceback_but_emits_fixed_safe_message(self):
        private_detail = "token=SETUP-SECRET at C:\\private\\setup.py:99"

        class SetupFailureController(FakeController):
            def enable_button_monitoring(self, _enabled):
                raise RuntimeError(private_detail)

        controller = SetupFailureController()
        events = []
        service = MakcuService(
            events.append,
            controller_factory=lambda **_kwargs: controller,
        )
        generation = service._begin_connection()

        with self.assertLogs("makcu_service", level="ERROR") as captured:
            service._connect_worker(generation)

        self.assertTrue(controller.disconnected)
        self.assertFalse(service.connected)
        self.assertIsNone(service.controller)
        self.assertEqual(
            events[-1],
            ServiceEvent(
                "disconnected",
                "Makcu unavailable; check USB and reconnect",
            ),
        )
        self.assertNotIn("SETUP-SECRET", str(events[-1].payload))
        self.assertNotIn("C:\\private", str(events[-1].payload))
        self.assertIn(private_detail, "\n".join(captured.output))
        self.assertTrue(any(record.exc_info is not None for record in captured.records))

    def test_old_generation_controller_is_disconnected_and_ignored(self):
        old = FakeController()
        factory_entered = threading.Event()
        release_factory = threading.Event()

        def factory(**_kwargs):
            factory_entered.set()
            if not release_factory.wait(1.0):
                raise TimeoutError("test did not release stale factory")
            return old

        service = MakcuService(lambda _event: None, controller_factory=factory)
        self.addCleanup(service.close)
        self.addCleanup(release_factory.set)
        generation = service._begin_connection()
        worker = threading.Thread(target=service._connect_worker, args=(generation,))
        worker.start()
        self.assertTrue(factory_entered.wait(1.0))
        service._begin_connection()
        release_factory.set()
        worker.join(1.0)

        self.assertFalse(worker.is_alive())
        self.assertTrue(old.disconnected)
        self.assertIsNone(service.controller)

    def test_worker_obsolete_before_factory_does_not_open_controller(self):
        factory_calls = 0

        def factory(**_kwargs):
            nonlocal factory_calls
            factory_calls += 1
            return FakeController()

        service = MakcuService(lambda _event: None, controller_factory=factory)
        obsolete_generation = service._begin_connection()
        service._begin_connection()

        service._connect_worker(obsolete_generation)

        self.assertEqual(factory_calls, 0)
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

    def test_current_button_callback_is_ignored_while_disconnected(self):
        controller = FakeController()
        events = []
        service = MakcuService(
            events.append,
            controller_factory=lambda **_kwargs: controller,
        )
        self.addCleanup(service.close)
        generation = service._begin_connection()
        service._connect_worker(generation)

        controller.connection_callback(False)
        self.assertEqual(events[-1], ServiceEvent("disconnected"))
        events_after_disconnect = list(events)

        controller.button_callback(MouseButton.LEFT, True)

        self.assertEqual(events, events_after_disconnect)

    def test_queued_button_callback_is_discarded_across_reconnect_transition(self):
        controller = FakeController()
        entered_connected_sink = threading.Event()
        release_connected_sink = threading.Event()
        events = []

        def sink(event):
            if event.kind == "connected":
                entered_connected_sink.set()
                self.assertTrue(release_connected_sink.wait(1.0))
            events.append(event)

        service = MakcuService(
            sink,
            controller_factory=lambda **_kwargs: controller,
        )
        self.addCleanup(service.close)
        generation = service._begin_connection()
        worker = threading.Thread(target=service._connect_worker, args=(generation,))
        worker.start()
        self.assertTrue(entered_connected_sink.wait(1.0))

        controller.button_callback(MouseButton.LEFT, True)
        controller.connection_callback(False)
        controller.connection_callback(True)
        release_connected_sink.set()
        worker.join(1.0)

        self.assertFalse(worker.is_alive())
        self.assertNotIn("button", [event.kind for event in events])

    def test_queued_connection_events_keep_only_the_latest_transition(self):
        controller = FakeController()
        entered_connected_sink = threading.Event()
        release_connected_sink = threading.Event()
        events = []

        def sink(event):
            if event.kind == "connected":
                entered_connected_sink.set()
                self.assertTrue(release_connected_sink.wait(1.0))
            events.append(event)

        service = MakcuService(
            sink,
            controller_factory=lambda **_kwargs: controller,
        )
        self.addCleanup(service.close)
        generation = service._begin_connection()
        worker = threading.Thread(target=service._connect_worker, args=(generation,))
        worker.start()
        self.assertTrue(entered_connected_sink.wait(1.0))

        controller.connection_callback(False)
        controller.connection_callback(True)
        controller.connection_callback(False)
        release_connected_sink.set()
        worker.join(1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(
            [event.kind for event in events],
            ["connecting", "connected", "disconnected"],
        )

    def test_queued_initial_connected_event_is_discarded_after_transition(self):
        controller = FakeController()
        entered_connecting_sink = threading.Event()
        release_connecting_sink = threading.Event()
        events = []

        def sink(event):
            if event.kind == "connecting":
                entered_connecting_sink.set()
                self.assertTrue(release_connecting_sink.wait(1.0))
            events.append(event)

        service = MakcuService(
            sink,
            controller_factory=lambda **_kwargs: controller,
        )
        self.addCleanup(service.close)
        begin_worker = threading.Thread(target=service._begin_connection)
        begin_worker.start()
        self.assertTrue(entered_connecting_sink.wait(1.0))
        generation = service.connection_generation
        connect_worker = threading.Thread(
            target=service._connect_worker,
            args=(generation,),
        )
        connect_worker.start()
        self.assertTrue(wait_until(lambda: service.controller is controller))

        controller.connection_callback(False)
        controller.connection_callback(True)
        release_connecting_sink.set()
        begin_worker.join(1.0)
        connect_worker.join(1.0)

        self.assertFalse(begin_worker.is_alive())
        self.assertFalse(connect_worker.is_alive())
        self.assertEqual(
            [event.kind for event in events],
            ["connecting", "reconnected"],
        )


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
        class OverflowDuration:
            def __float__(self):
                raise OverflowError("duration overflow")

        disconnected = MakcuService(lambda _event: None)
        self.addCleanup(disconnected.close)
        self.assertFalse(
            disconnected.start_ai_motion(lambda: None, AimSettings)
        )

        invalid_durations = (
            object(),
            "not-a-duration",
            float("inf"),
            float("-inf"),
            float("nan"),
            OverflowDuration(),
        )
        for invalid_duration in invalid_durations:
            with self.subTest(duration=invalid_duration):
                service, _controller, _events = self.connected_service()
                try:
                    accepted = service.start_ai_motion(
                        lambda: None, AimSettings, invalid_duration
                    )
                except Exception as exc:
                    self.fail(f"invalid duration raised {type(exc).__name__}: {exc}")
                finally:
                    service.stop_motion("test_cleanup")
                    service.join_motion(1.0)
                self.assertFalse(accepted)

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

    def test_jitter_and_ai_duration_terminals_carry_the_returned_source(self):
        for mode in ("jitter", "ai"):
            with self.subTest(mode=mode):
                service, _controller, events = self.connected_service()
                events.clear()

                if mode == "ai":
                    start_with_source = getattr(
                        service,
                        "start_ai_motion_source",
                        None,
                    )
                    self.assertIsNotNone(start_with_source)
                    source = start_with_source(
                        lambda: None,
                        AimSettings,
                        duration_s=0,
                    )
                else:
                    start_with_source = getattr(
                        service,
                        "start_motion_source",
                        None,
                    )
                    self.assertIsNotNone(start_with_source)
                    source = start_with_source(
                        lambda: MotionSettings(),
                        duration_s=0,
                    )
                service.join_motion(1.0)

                self.assertIs(type(source), int)
                self.assertGreater(source, 0)
                self.assertEqual(events[-1].kind, "motion_stopped")
                self.assertEqual(events[-1].payload, "duration_complete")
                self.assertEqual(
                    getattr(events[-1], "motion_generation", None),
                    source,
                )

    def test_reentrant_same_reason_motion_starts_keep_distinct_sources(self):
        for mode in ("jitter", "ai"):
            with self.subTest(mode=mode):
                terminal_events = []
                reentrant_sources = []
                two_terminals = threading.Event()
                controller = FakeController()
                service = None

                def start_timed_motion():
                    if mode == "ai":
                        start_with_source = getattr(
                            service,
                            "start_ai_motion_source",
                            None,
                        )
                        self.assertIsNotNone(start_with_source)
                        return start_with_source(
                            lambda: None,
                            AimSettings,
                            duration_s=0,
                        )
                    start_with_source = getattr(
                        service,
                        "start_motion_source",
                        None,
                    )
                    self.assertIsNotNone(start_with_source)
                    return start_with_source(
                        lambda: MotionSettings(),
                        duration_s=0,
                    )

                def sink(event):
                    if (
                        event.kind != "motion_stopped"
                        or event.payload != "duration_complete"
                    ):
                        return
                    terminal_events.append(event)
                    if len(terminal_events) == 1:
                        reentrant_sources.append(start_timed_motion())
                    if len(terminal_events) == 2:
                        two_terminals.set()

                service = MakcuService(
                    sink,
                    controller_factory=lambda **_kwargs: controller,
                    engine_factory=RecordingEngine,
                    aim_engine_factory=FakeAimEngine,
                )
                self.addCleanup(service.close)
                service._connect_worker(service._begin_connection())

                first_source = start_timed_motion()
                self.assertTrue(two_terminals.wait(1.0))
                service.join_motion(1.0)

                self.assertEqual(len(reentrant_sources), 1)
                second_source = reentrant_sources[0]
                self.assertIs(type(first_source), int)
                self.assertIs(type(second_source), int)
                self.assertNotEqual(first_source, second_source)
                self.assertEqual(
                    [
                        getattr(event, "motion_generation", None)
                        for event in terminal_events
                    ],
                    [first_source, second_source],
                )

    def test_jitter_and_ai_motion_errors_carry_the_returned_source(self):
        for mode in ("jitter", "ai"):
            with self.subTest(mode=mode):
                service, _controller, events = self.connected_service()
                events.clear()

                def fail():
                    raise RuntimeError(f"{mode} source failure")

                if mode == "ai":
                    start_with_source = getattr(
                        service,
                        "start_ai_motion_source",
                        None,
                    )
                    self.assertIsNotNone(start_with_source)
                    source = start_with_source(fail, AimSettings)
                else:
                    start_with_source = getattr(
                        service,
                        "start_motion_source",
                        None,
                    )
                    self.assertIsNotNone(start_with_source)
                    source = start_with_source(fail)
                service.join_motion(1.0)

                self.assertIs(type(source), int)
                self.assertEqual(events[-1].kind, "motion_error")
                self.assertEqual(
                    getattr(events[-1], "motion_generation", None),
                    source,
                )

    def test_ai_duration_crossed_during_blocked_step_prevents_move(self):
        engine = BlockingAimEngine()
        deadline_crossed = threading.Event()
        target = TargetSnapshot(1, 0.0, "head", 170, 160)
        service, controller, events = self.connected_service(
            aim_engine_factory=lambda: engine
        )
        self.addCleanup(engine.release.set)

        def clock():
            return 1.0 if deadline_crossed.is_set() else 0.0

        with mock.patch("makcu_service.time.perf_counter", side_effect=clock):
            self.assertTrue(
                service.start_ai_motion(
                    lambda: target, AimSettings, duration_s=0.5
                )
            )
            self.assertTrue(engine.entered.wait(1.0))
            deadline_crossed.set()
            engine.release.set()
            service.join_motion(1.0)

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

    def test_disconnect_timeout_suppresses_late_ai_error_event(self):
        engine = BlockingFailingAimEngine()
        target = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)
        service, _controller, events = self.connected_service(
            aim_engine_factory=lambda: engine
        )
        self.addCleanup(engine.release.set)
        self.assertTrue(service.start_ai_motion(lambda: target, AimSettings))
        self.assertTrue(engine.entered.wait(1.0))

        with mock.patch.object(service, "join_motion", return_value=None):
            service._connection_changed(service.connection_generation, False)
        events_after_disconnect = len(events)
        engine.release.set()
        service.join_motion(1.0)

        self.assertEqual(events[events_after_disconnect:], [])

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

    def test_terminal_callback_does_not_hold_service_state_lock(self):
        callback_entered = threading.Event()
        reader_started = threading.Event()
        reader_finished = threading.Event()
        callback_observed_reader = threading.Event()
        callback_finished = threading.Event()
        reader_threads = []
        service = None

        def sink(event):
            if event.kind != "motion_stopped":
                return
            callback_entered.set()

            def read_property():
                reader_started.set()
                service.connected
                reader_finished.set()

            reader = threading.Thread(target=read_property)
            reader_threads.append(reader)
            reader.start()
            if reader_started.wait(1.0) and reader_finished.wait(1.0):
                callback_observed_reader.set()
            callback_finished.set()

        controller = FakeController()
        service = MakcuService(
            sink,
            controller_factory=lambda **_kwargs: controller,
            aim_engine_factory=FakeAimEngine,
        )
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())

        self.assertTrue(
            service.start_ai_motion(lambda: None, AimSettings, duration_s=0)
        )
        self.assertTrue(callback_entered.wait(1.0))
        self.assertTrue(callback_finished.wait(2.0))
        service.join_motion(2.0)
        for reader in reader_threads:
            reader.join(1.0)

        self.assertTrue(callback_observed_reader.is_set())

    def test_new_motion_waits_for_prior_terminal_callback(self):
        terminal_entered = threading.Event()
        release_terminal = threading.Event()
        terminal_delivered = threading.Event()
        starter_entered = threading.Event()
        start_returned = threading.Event()
        start_observed_terminal = threading.Event()
        start_results = []
        blocker = BlockingAimEngine()

        def sink(event):
            if event.kind != "motion_stopped" or terminal_entered.is_set():
                return
            terminal_entered.set()
            release_terminal.wait(1.0)
            terminal_delivered.set()

        controller = FakeController()
        service = MakcuService(
            sink,
            controller_factory=lambda **_kwargs: controller,
            aim_engine_factory=lambda: blocker,
        )
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())
        target = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)
        self.assertTrue(
            service.start_ai_motion(lambda: None, AimSettings, duration_s=0)
        )
        self.assertTrue(terminal_entered.wait(1.0))

        def start_next_motion():
            starter_entered.set()
            start_results.append(
                service.start_ai_motion(lambda: target, AimSettings)
            )
            if terminal_delivered.is_set():
                start_observed_terminal.set()
            start_returned.set()

        starter = threading.Thread(target=start_next_motion)
        starter.start()
        try:
            self.assertTrue(starter_entered.wait(1.0))
            self.assertFalse(start_returned.wait(0.05))
            release_terminal.set()
            self.assertTrue(start_returned.wait(1.0))
            self.assertTrue(blocker.entered.wait(1.0))
        finally:
            release_terminal.set()
            service.stop_motion("test_cleanup")
            blocker.release.set()
            starter.join(1.0)
            service.join_motion(1.0)

        self.assertEqual(start_results, [True])
        self.assertTrue(start_observed_terminal.is_set())

    def test_stop_invalidates_start_waiting_for_terminal_dispatch(self):
        class MoveObservedController(FakeController):
            def __init__(self):
                super().__init__()
                self.move_observed = threading.Event()

            def move(self, x, y):
                super().move(x, y)
                self.move_observed.set()

        terminal_entered = threading.Event()
        release_terminal = threading.Event()
        dispatch_wait_entered = threading.Event()
        start_returned = threading.Event()
        start_results = []
        engine = BlockingAimEngine()

        def sink(event):
            if event.kind != "motion_stopped" or terminal_entered.is_set():
                return
            terminal_entered.set()
            release_terminal.wait(1.0)

        controller = MoveObservedController()
        service = MakcuService(
            sink,
            controller_factory=lambda **_kwargs: controller,
            aim_engine_factory=lambda: engine,
        )
        self.addCleanup(engine.release.set)
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())
        target = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)
        self.assertTrue(
            service.start_ai_motion(lambda: None, AimSettings, duration_s=0)
        )
        self.assertTrue(terminal_entered.wait(1.0))
        real_dispatch_wait = service._motion_dispatch_done.wait

        def observed_dispatch_wait(timeout=None):
            dispatch_wait_entered.set()
            return real_dispatch_wait(timeout)

        def start_next_motion():
            start_results.append(
                service.start_ai_motion(lambda: target, AimSettings)
            )
            start_returned.set()

        starter = threading.Thread(target=start_next_motion)
        with mock.patch.object(
            service._motion_dispatch_done,
            "wait",
            side_effect=observed_dispatch_wait,
        ):
            starter.start()
            try:
                self.assertTrue(dispatch_wait_entered.wait(1.0))
                service.stop_motion("manual")
                engine.release.set()
                release_terminal.set()
                self.assertTrue(start_returned.wait(1.0))
                if start_results == [True]:
                    self.assertTrue(controller.move_observed.wait(1.0))
                    service.stop_motion("test_cleanup")
                service.join_motion(1.0)
            finally:
                release_terminal.set()
                engine.release.set()
                service.stop_motion("test_cleanup")
                starter.join(1.0)
                service.join_motion(1.0)

        self.assertEqual((start_results, controller.moves), ([False], []))

    def test_terminal_callback_can_start_next_motion_directly(self):
        callback_finished = threading.Event()
        reentrant_attempted = threading.Event()
        start_results = []
        blocker = BlockingAimEngine()
        target = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)
        service = None

        def sink(event):
            if event.kind != "motion_stopped" or reentrant_attempted.is_set():
                return
            reentrant_attempted.set()
            start_results.append(
                service.start_ai_motion(lambda: target, AimSettings)
            )
            callback_finished.set()

        controller = FakeController()
        service = MakcuService(
            sink,
            controller_factory=lambda **_kwargs: controller,
            aim_engine_factory=lambda: blocker,
        )
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())

        self.assertTrue(
            service.start_ai_motion(lambda: None, AimSettings, duration_s=0)
        )
        self.assertTrue(callback_finished.wait(1.0))
        self.assertEqual(start_results, [True])
        self.assertTrue(blocker.entered.wait(1.0))

        service.stop_motion("test_cleanup")
        blocker.release.set()
        service.join_motion(1.0)

    def test_close_suppresses_terminal_event_queued_behind_callback(self):
        first_callback_entered = threading.Event()
        second_terminal_reserved = threading.Event()
        release_first_callback = threading.Event()
        first_callback_finished = threading.Event()
        events = []
        service = None

        def sink(event):
            events.append(event)
            if event.kind != "motion_stopped" or first_callback_entered.is_set():
                return
            first_callback_entered.set()
            service.start_ai_motion(lambda: None, AimSettings, duration_s=0)
            service.join_motion(1.0)
            second_terminal_reserved.set()
            release_first_callback.wait(1.0)
            first_callback_finished.set()

        controller = FakeController()
        service = MakcuService(
            sink,
            controller_factory=lambda **_kwargs: controller,
        )
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())
        events.clear()
        self.assertTrue(
            service.start_ai_motion(lambda: None, AimSettings, duration_s=0)
        )
        self.assertTrue(first_callback_entered.wait(1.0))
        self.assertTrue(second_terminal_reserved.wait(1.0))

        service.close()
        release_first_callback.set()
        self.assertTrue(first_callback_finished.wait(1.0))
        service.join_motion(1.0)

        self.assertEqual(
            events, [ServiceEvent("motion_stopped", "duration_complete")]
        )

    def test_close_suppresses_queued_reconnect_events(self):
        terminal_entered = threading.Event()
        release_terminal = threading.Event()
        terminal_finished = threading.Event()
        events = []

        def sink(event):
            events.append(event)
            if event.kind != "motion_stopped":
                return
            terminal_entered.set()
            release_terminal.wait(1.0)
            terminal_finished.set()

        old = FakeController()
        fresh = FakeController()
        controllers = iter((old, fresh))
        service = MakcuService(
            sink,
            controller_factory=lambda **_kwargs: next(controllers),
        )
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())
        events.clear()
        self.assertTrue(
            service.start_ai_motion(lambda: None, AimSettings, duration_s=0)
        )
        self.assertTrue(terminal_entered.wait(1.0))

        service.reconnect()
        self.assertTrue(wait_until(lambda: service.controller is fresh))
        service.close()
        release_terminal.set()
        self.assertTrue(terminal_finished.wait(1.0))
        service.join_motion(1.0)

        self.assertEqual(
            events, [ServiceEvent("motion_stopped", "duration_complete")]
        )

    def test_terminal_callback_helper_can_close_service(self):
        callback_entered = threading.Event()
        helper_finished = threading.Event()
        callback_observed_helper = threading.Event()
        callback_finished = threading.Event()
        helper_threads = []
        service = None

        def sink(event):
            if event.kind != "motion_stopped" or callback_entered.is_set():
                return
            callback_entered.set()

            def close_service():
                service.close()
                helper_finished.set()

            helper = threading.Thread(target=close_service)
            helper_threads.append(helper)
            helper.start()
            if helper_finished.wait(0.25):
                callback_observed_helper.set()
            callback_finished.set()

        controller = FakeController()
        service = MakcuService(
            sink,
            controller_factory=lambda **_kwargs: controller,
        )
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())

        self.assertTrue(
            service.start_ai_motion(lambda: None, AimSettings, duration_s=0)
        )
        self.assertTrue(callback_finished.wait(1.0))
        for helper in helper_threads:
            helper.join(1.0)

        self.assertTrue(callback_observed_helper.is_set())
        self.assertTrue(helper_finished.is_set())
        self.assertFalse(service.connected)

    def test_terminal_callback_helper_can_reconnect_service(self):
        callback_entered = threading.Event()
        helper_finished = threading.Event()
        callback_observed_helper = threading.Event()
        callback_finished = threading.Event()
        helper_threads = []
        old = FakeController()
        fresh = FakeController()
        controllers = iter((old, fresh))
        service = None

        def sink(event):
            if event.kind != "motion_stopped" or callback_entered.is_set():
                return
            callback_entered.set()

            def reconnect_service():
                service.reconnect()
                helper_finished.set()

            helper = threading.Thread(target=reconnect_service)
            helper_threads.append(helper)
            helper.start()
            if helper_finished.wait(0.25):
                callback_observed_helper.set()
            callback_finished.set()

        service = MakcuService(
            sink,
            controller_factory=lambda **_kwargs: next(controllers),
        )
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())

        self.assertTrue(
            service.start_ai_motion(lambda: None, AimSettings, duration_s=0)
        )
        self.assertTrue(callback_finished.wait(1.0))
        for helper in helper_threads:
            helper.join(1.0)

        self.assertTrue(callback_observed_helper.is_set())
        self.assertTrue(helper_finished.is_set())
        self.assertTrue(wait_until(lambda: service.controller is fresh))

    def test_join_motion_waits_for_terminal_callback(self):
        terminal_entered = threading.Event()
        release_terminal = threading.Event()
        terminal_finished = threading.Event()
        join_entered = threading.Event()
        join_returned = threading.Event()
        join_observed_terminal = threading.Event()

        def sink(event):
            if event.kind != "motion_stopped":
                return
            terminal_entered.set()
            release_terminal.wait(1.0)
            terminal_finished.set()

        controller = FakeController()
        service = MakcuService(
            sink,
            controller_factory=lambda **_kwargs: controller,
        )
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())
        self.assertTrue(
            service.start_ai_motion(lambda: None, AimSettings, duration_s=0)
        )
        self.assertTrue(terminal_entered.wait(1.0))

        def join_motion():
            join_entered.set()
            service.join_motion()
            if terminal_finished.is_set():
                join_observed_terminal.set()
            join_returned.set()

        joiner = threading.Thread(target=join_motion)
        joiner.start()
        try:
            self.assertTrue(join_entered.wait(1.0))
            self.assertFalse(join_returned.wait(0.05))
            release_terminal.set()
            self.assertTrue(join_returned.wait(1.0))
        finally:
            release_terminal.set()
            joiner.join(1.0)

        self.assertTrue(join_observed_terminal.is_set())

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

    def test_close_suppresses_stale_ai_error_event(self):
        engine = BlockingFailingAimEngine()
        target = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)
        service, _controller, events = self.connected_service(
            aim_engine_factory=lambda: engine
        )
        self.addCleanup(engine.release.set)
        self.assertTrue(service.start_ai_motion(lambda: target, AimSettings))
        self.assertTrue(engine.entered.wait(1.0))

        service.close()
        events_after_close = len(events)
        engine.release.set()
        service.join_motion(1.0)

        self.assertEqual(events[events_after_close:], [])

    def test_reconnect_suppresses_stale_ai_error_event(self):
        engine = BlockingFailingAimEngine()
        old = FakeController()
        fresh = FakeController()
        controllers = iter((old, fresh))
        events = []
        service = MakcuService(
            events.append,
            controller_factory=lambda **_kwargs: next(controllers),
            aim_engine_factory=lambda: engine,
        )
        self.addCleanup(engine.release.set)
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())
        target = TargetSnapshot(1, time.perf_counter(), "head", 170, 160)
        self.assertTrue(service.start_ai_motion(lambda: target, AimSettings))
        self.assertTrue(engine.entered.wait(1.0))

        service.reconnect()
        self.assertTrue(wait_until(lambda: service.controller is fresh))
        service.join_connection(1.0)
        events_after_reconnect = len(events)
        engine.release.set()
        service.join_motion(1.0)

        self.assertEqual(events[events_after_reconnect:], [])

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

    def test_paired_pulse_worker_sends_diagonal_reports_and_stop_prevents_next_half(self):
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
            all(x > 0 and y < 0 and abs(x) <= 2 and abs(y) <= 2
                for x, y in moves_after_stop)
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
                lambda: MotionSettings(2, 120, "Instant"),
                None,
            )
        self.assertEqual(stop_event.timeouts, [1 / 240])

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

    def test_cancel_motion_returns_while_move_is_blocked_and_stops_next_report(self):
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
        service = MakcuService(
            lambda _event: None,
            controller_factory=lambda **_kwargs: controller,
            engine_factory=RecordingEngine,
        )
        self.addCleanup(controller.release_move.set)
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())
        self.assertTrue(service.start_motion(lambda: MotionSettings()))
        self.assertTrue(controller.move_entered.wait(1.0))

        cancel_returned = threading.Event()
        cancel_thread = threading.Thread(
            target=lambda: (
                service.cancel_motion("nonblocking_stop"),
                cancel_returned.set(),
            )
        )
        cancel_thread.start()
        try:
            self.assertTrue(cancel_returned.wait(0.2))
            self.assertTrue(service._motion_stop.is_set())
            self.assertEqual(controller.move_starts, 1)
            controller.release_move.set()
            service.join_motion(1.0)
        finally:
            controller.release_move.set()
            cancel_thread.join(1.0)

        self.assertFalse(service.motion_active)
        self.assertEqual(controller.move_starts, 1)

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
    def test_health_thread_start_failure_clears_owned_thread_reference(self):
        class FailingThread:
            def start(self):
                raise RuntimeError("thread start failed")

        controller = FakeController()
        service = MakcuService(
            lambda _event: None,
            controller_factory=lambda **_kwargs: controller,
        )
        self.addCleanup(service.close)
        generation = service._begin_connection()

        with mock.patch(
            "makcu_service.threading.Thread",
            return_value=FailingThread(),
        ), self.assertLogs("makcu_service", level="ERROR"):
            service._connect_worker(generation)

        self.assertIsNone(service._health_thread)
        self.assertTrue(controller.disconnected)

    def test_health_worker_start_failure_disconnects_exact_controller(self):
        controller = FakeController()
        events = []
        service = MakcuService(
            events.append,
            controller_factory=lambda **_kwargs: controller,
        )
        self.addCleanup(service.close)

        with mock.patch.object(
            service,
            "_start_health_worker",
            side_effect=RuntimeError("health thread could not start"),
        ), self.assertLogs("makcu_service", level="ERROR"):
            service.connect()
            service.join_connection(1.0)

        self.assertTrue(controller.disconnected)
        self.assertFalse(service.connected)
        self.assertIsNone(service.controller)
        self.assertNotIn("connected", [event.kind for event in events])

    def test_reconnect_during_health_start_rollback_disconnects_old_once(self):
        health_start_entered = threading.Event()
        release_health_start = threading.Event()
        factory_calls = 0

        class SecondDisconnectFails(FakeController):
            def __init__(self):
                super().__init__()
                self.disconnect_calls = 0

            def disconnect(self):
                self.disconnect_calls += 1
                if self.disconnect_calls > 1:
                    raise RuntimeError("controller disconnected twice")
                super().disconnect()

        old = SecondDisconnectFails()
        fresh = FakeController()

        def factory(**_kwargs):
            nonlocal factory_calls
            factory_calls += 1
            return old if factory_calls == 1 else fresh

        service = MakcuService(lambda _event: None, controller_factory=factory)
        self.addCleanup(release_health_start.set)
        self.addCleanup(service.close)
        original_start_health = service._start_health_worker

        def fail_first_health_start(generation, controller):
            if controller is old:
                health_start_entered.set()
                if not release_health_start.wait(1.0):
                    raise TimeoutError("test did not release health start")
                raise RuntimeError("health thread could not start")
            return original_start_health(generation, controller)

        with mock.patch.object(
            service,
            "_start_health_worker",
            side_effect=fail_first_health_start,
        ), self.assertLogs("makcu_service", level="ERROR"):
            service.connect()
            self.assertTrue(health_start_entered.wait(1.0))
            service.reconnect()
            release_health_start.set()
            service.join_connection(1.0)

        self.assertEqual(old.disconnect_calls, 1)
        self.assertEqual(factory_calls, 2)
        self.assertIs(service.controller, fresh)
        self.assertTrue(service.connected)

    def test_reconnect_returns_while_move_is_blocked_then_releases_port_first(self):
        order = []

        class BlockingMoveController(FakeController):
            def __init__(self):
                super().__init__()
                self.move_entered = threading.Event()
                self.release_move = threading.Event()

            def move(self, x, y):
                order.append("move_entered")
                self.move_entered.set()
                if not self.release_move.wait(2.0):
                    raise TimeoutError("test did not release blocked move")
                order.append("move_returned")
                super().move(x, y)

            def disconnect(self):
                order.append("old_disconnected")
                super().disconnect()

        old = BlockingMoveController()
        fresh = FakeController()
        factory_calls = 0

        def factory(**_kwargs):
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == 1:
                return old
            order.append("replacement_factory")
            return fresh

        service = MakcuService(
            lambda _event: None,
            controller_factory=factory,
            engine_factory=RecordingEngine,
        )
        self.addCleanup(old.release_move.set)
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())
        self.assertTrue(service.start_motion(lambda: MotionSettings()))
        self.assertTrue(old.move_entered.wait(1.0))

        reconnect_returned = threading.Event()
        reconnect_thread = threading.Thread(
            target=lambda: (service.reconnect(), reconnect_returned.set())
        )
        reconnect_thread.start()
        try:
            self.assertTrue(reconnect_returned.wait(1.0))
            self.assertFalse(service.connected)
            old.release_move.set()
            self.assertTrue(wait_until(lambda: service.controller is fresh))
            service.join_connection(1.0)
        finally:
            old.release_move.set()
            reconnect_thread.join(1.0)
            service.join_motion(1.0)

        self.assertEqual(factory_calls, 2)
        self.assertLess(order.index("move_returned"), order.index("old_disconnected"))
        self.assertLess(
            order.index("old_disconnected"), order.index("replacement_factory")
        )

    def test_recovery_during_disconnect_stop_rearms_the_next_loss(self):
        class BlockingMoveController(FakeController):
            def __init__(self):
                super().__init__()
                self.move_entered = threading.Event()
                self.release_move = threading.Event()

            def move(self, x, y):
                self.move_entered.set()
                if not self.release_move.wait(2.0):
                    raise TimeoutError("test did not release blocked move")
                super().move(x, y)

        controller = BlockingMoveController()
        events = []
        service = MakcuService(
            events.append,
            controller_factory=lambda **_kwargs: controller,
            engine_factory=RecordingEngine,
        )
        self.addCleanup(controller.release_move.set)
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())
        self.assertTrue(service.start_motion(lambda: MotionSettings()))
        self.assertTrue(controller.move_entered.wait(1.0))
        first_loss = threading.Thread(
            target=lambda: controller.connection_callback(False)
        )

        first_loss.start()
        self.assertTrue(service._motion_stop.wait(1.0))
        controller.connection_callback(True)
        controller.release_move.set()
        first_loss.join(1.0)
        controller.connection_callback(False)
        service.join_motion(1.0)

        self.assertFalse(first_loss.is_alive())
        self.assertEqual(
            [
                event.kind
                for event in events
                if event.kind
                in {"connecting", "connected", "reconnected", "disconnected"}
            ],
            ["connecting", "connected", "reconnected", "disconnected"],
        )

    def test_repeated_reconnects_use_latest_generation_after_one_teardown(self):
        class BlockingDisconnectController(FakeController):
            def __init__(self):
                super().__init__()
                self.disconnect_entered = threading.Event()
                self.release_disconnect = threading.Event()
                self.disconnect_calls = 0

            def disconnect(self):
                self.disconnect_calls += 1
                self.disconnect_entered.set()
                if not self.release_disconnect.wait(2.0):
                    raise TimeoutError("test did not release old controller")
                super().disconnect()

        old = BlockingDisconnectController()
        fresh = FakeController()
        factory_calls = 0

        def factory(**_kwargs):
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == 1:
                return old
            return fresh

        events = []
        service = MakcuService(events.append, controller_factory=factory)
        self.addCleanup(old.release_disconnect.set)
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())
        events.clear()

        first_generation = service.reconnect()
        self.assertTrue(old.disconnect_entered.wait(1.0))
        second_generation = service.reconnect()
        latest_generation = service.reconnect()
        requested_worker_names = {
            f"MakcuConnect-{generation}"
            for generation in (
                first_generation,
                second_generation,
                latest_generation,
            )
        }
        self.assertEqual(
            len([
                thread
                for thread in threading.enumerate()
                if thread.name in requested_worker_names and thread.is_alive()
            ]),
            1,
        )
        old.release_disconnect.set()

        self.assertTrue(wait_until(lambda: service.controller is fresh))
        service.join_connection(1.0)
        self.assertEqual(
            [first_generation, second_generation, latest_generation],
            sorted({first_generation, second_generation, latest_generation}),
        )
        self.assertEqual(service.connection_generation, latest_generation)
        self.assertEqual(old.disconnect_calls, 1)
        self.assertEqual(factory_calls, 2)
        self.assertEqual(
            [event.kind for event in events],
            ["connecting", "connecting", "connecting", "connected"],
        )

    def test_old_disconnect_failure_suppresses_factory_until_explicit_retry(self):
        private_detail = "password=PORT-SECRET at C:\\private\\disconnect.py:23"

        class DisconnectFailureController(FakeController):
            def __init__(self):
                super().__init__()
                self.disconnect_entered = threading.Event()
                self.release_disconnect = threading.Event()
                self.disconnect_calls = 0

            def disconnect(self):
                self.disconnect_calls += 1
                self.disconnect_entered.set()
                if not self.release_disconnect.wait(2.0):
                    raise TimeoutError("test did not release old controller")
                if self.disconnect_calls == 1:
                    raise RuntimeError(private_detail)
                super().disconnect()

        old = DisconnectFailureController()
        fresh = FakeController()
        events = []
        factory_calls = 0

        def factory(**_kwargs):
            nonlocal factory_calls
            factory_calls += 1
            return old if factory_calls == 1 else fresh

        service = MakcuService(
            events.append,
            controller_factory=factory,
        )
        self.addCleanup(old.release_disconnect.set)
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())

        with self.assertLogs("makcu_service", level="ERROR") as captured:
            service.reconnect()
            self.assertTrue(old.disconnect_entered.wait(1.0))
            failed_generation = service.reconnect()
            old.release_disconnect.set()
            service.join_connection(1.0)

        self.assertFalse(service.connected)
        self.assertIsNone(service.controller)
        self.assertEqual(service.connection_generation, failed_generation)
        self.assertEqual(factory_calls, 1)
        self.assertEqual(
            events[-1],
            ServiceEvent(
                "disconnected",
                "Makcu unavailable; check USB and reconnect",
            ),
        )
        self.assertFalse(any(
            "PORT-SECRET" in str(event.payload) or "C:\\private" in str(event.payload)
            for event in events
        ))
        self.assertIn(private_detail, "\n".join(captured.output))
        self.assertTrue(any(record.exc_info is not None for record in captured.records))

        service.reconnect()
        self.assertTrue(wait_until(lambda: service.controller is fresh))
        service.join_connection(1.0)
        self.assertEqual(old.disconnect_calls, 2)
        self.assertEqual(factory_calls, 2)
        self.assertTrue(service.connected)

    def test_close_cancels_reconnects_waiting_behind_blocked_teardown(self):
        class BlockingDisconnectController(FakeController):
            def __init__(self):
                super().__init__()
                self.disconnect_started = threading.Event()
                self.release_disconnect = threading.Event()

            def disconnect(self):
                self.disconnect_started.set()
                if not self.release_disconnect.wait(1.0):
                    raise TimeoutError("test did not release old controller")
                super().disconnect()

        old = BlockingDisconnectController()
        factory_calls = 0

        def factory(**_kwargs):
            nonlocal factory_calls
            factory_calls += 1
            return old if factory_calls == 1 else FakeController()

        service = MakcuService(lambda _event: None, controller_factory=factory)
        self.addCleanup(old.release_disconnect.set)
        service._connect_worker(service._begin_connection())

        service.reconnect()
        self.assertTrue(old.disconnect_started.wait(1.0))
        service.reconnect()
        service.reconnect()
        service.close()
        old.release_disconnect.set()
        service.join_connection(1.0)

        self.assertTrue(old.disconnected)
        self.assertEqual(factory_calls, 1)
        self.assertFalse(service.connected)
        self.assertIsNone(service.controller)

    def test_close_wakes_health_monitor_without_waiting_for_poll_interval(self):
        controller = FakeController()
        service = MakcuService(
            lambda _event: None,
            controller_factory=lambda **_kwargs: controller,
        )
        service._connect_worker(service._begin_connection())
        health_thread = service._health_thread
        self.assertIsNotNone(health_thread)
        self.assertTrue(health_thread.is_alive())

        service.close()
        health_thread.join(0.05)

        self.assertFalse(health_thread.is_alive())

    def test_reconnect_wakes_obsolete_health_monitor_immediately(self):
        old = FakeController()
        fresh = FakeController()
        controllers = iter((old, fresh))
        service = MakcuService(
            lambda _event: None,
            controller_factory=lambda **_kwargs: next(controllers),
        )
        self.addCleanup(service.close)
        service._connect_worker(service._begin_connection())
        old_health_thread = service._health_thread
        self.assertIsNotNone(old_health_thread)
        self.assertTrue(old_health_thread.is_alive())

        service.reconnect()
        old_health_thread.join(0.05)

        self.assertFalse(old_health_thread.is_alive())
        self.assertTrue(wait_until(lambda: service.controller is fresh))

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
        self.assertTrue(wait_until(lambda: service.controller is fresh))

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
