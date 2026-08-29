import tempfile
import threading
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from jitter_app.ai.detection import ModelContractError
from jitter_app.ai.model_selection import (
    ModelChoice,
    ModelSelectionError,
    ModelValidationEvent,
    ModelValidator,
    bundled_model_choice,
    external_model_choice,
)


class ModelChoiceTests(unittest.TestCase):
    def test_bundled_choice_has_known_320_input_size(self):
        with mock.patch(
            "jitter_app.ai.model_selection.model_resource_path",
            return_value=Path("models/all_games_320.onnx"),
        ):
            choice = bundled_model_choice()
        self.assertEqual(choice.input_size, 320)

    def test_bundled_choice_uses_resolved_default_resource(self):
        with mock.patch(
            "jitter_app.ai.model_selection.model_resource_path",
            return_value=Path("models/all_games_320.onnx"),
        ):
            choice = bundled_model_choice()
        self.assertEqual(choice.path.name, "all_games_320.onnx")
        self.assertTrue(choice.path.is_absolute())
        self.assertEqual(choice.display_name, "all_games_320.onnx")
        self.assertTrue(choice.is_default)
        with self.assertRaises(FrozenInstanceError):
            choice.display_name = "changed.onnx"

    def test_external_choice_has_no_trusted_size_before_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.onnx"
            path.write_bytes(b"model")
            choice = external_model_choice(path)
        self.assertIsNone(choice.input_size)

    def test_external_choice_accepts_case_insensitive_onnx_and_resolves_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.ONNX"
            path.write_bytes(b"model")
            choice = external_model_choice(path)
        self.assertEqual(choice.path, path.resolve())
        self.assertEqual(choice.display_name, "custom.ONNX")
        self.assertFalse(choice.is_default)

    def test_external_choice_rejects_missing_wrong_suffix_and_directory_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = root / "custom.txt"
            wrong.write_bytes(b"model")
            cases = (
                (root / "missing.onnx", "Selected model file was not found"),
                (wrong, "Select an ONNX model file"),
                (root, "Selected model must be a file"),
            )
            for raw_path, message in cases:
                with self.subTest(raw_path=raw_path):
                    with self.assertRaisesRegex(ModelSelectionError, message):
                        external_model_choice(raw_path)


class ModelValidatorTests(unittest.TestCase):
    def test_validator_constructs_model_on_named_daemon_without_inference(self):
        events = []
        finished = threading.Event()
        calls = []

        class ContractOnlyDetector:
            provider = "DmlExecutionProvider"
            input_size = 320

        def detector_factory(path):
            thread = threading.current_thread()
            calls.append((path, thread.name, thread.daemon))
            return ContractOnlyDetector()

        choice = ModelChoice(Path("chosen.onnx"), "chosen.onnx", False)
        validator = ModelValidator(
            lambda event: (events.append(event), finished.set()),
            detector_factory=detector_factory,
        )
        self.addCleanup(validator.close)
        self.assertTrue(validator.start(choice, 4))
        self.assertTrue(finished.wait(1.0))
        self.assertEqual(calls, [(choice.path, "ModelValidation-4", True)])
        self.assertEqual(
            events,
            [ModelValidationEvent("ready", 4, ModelChoice(
                choice.path, choice.display_name, choice.is_default, 320
            ))],
        )

    def test_validator_publishes_choice_enriched_with_detector_input_size(self):
        events = []
        finished = threading.Event()

        class Detector:
            provider = "DmlExecutionProvider"
            input_size = 640

        choice = ModelChoice(Path("chosen.onnx"), "chosen.onnx", False)
        validator = ModelValidator(
            lambda event: (events.append(event), finished.set()),
            detector_factory=lambda _path: Detector(),
        )
        self.addCleanup(validator.close)
        self.assertTrue(validator.start(choice, 4))
        self.assertTrue(finished.wait(1.0))
        self.assertEqual(events[0].choice.input_size, 640)
        self.assertEqual(events[0].choice.path, choice.path)
        self.assertIsNone(choice.input_size)

    def test_new_validation_cancels_late_result_from_previous_token(self):
        first_entered = threading.Event()
        release_first = threading.Event()
        first_returned = threading.Event()
        second_ready = threading.Event()
        events = []

        class Detector:
            provider = "CPUExecutionProvider"
            input_size = 160

        def detector_factory(path):
            if path.name == "first.onnx":
                first_entered.set()
                release_first.wait(1.0)
                first_returned.set()
            return Detector()

        def sink(event):
            events.append(event)
            if event.token == 2:
                second_ready.set()

        validator = ModelValidator(sink, detector_factory=detector_factory)
        self.addCleanup(validator.close)
        first = ModelChoice(Path("first.onnx"), "first.onnx", False)
        second = ModelChoice(Path("second.onnx"), "second.onnx", False)
        self.assertTrue(validator.start(first, 1))
        self.assertTrue(first_entered.wait(1.0))
        self.assertTrue(validator.start(second, 2))
        self.assertTrue(second_ready.wait(1.0))
        release_first.set()
        self.assertTrue(first_returned.wait(1.0))
        self.assertEqual(
            events,
            [ModelValidationEvent("ready", 2, ModelChoice(
                second.path, second.display_name, second.is_default, 160
            ))],
        )

    def test_cancel_cannot_transition_during_current_event_publication(self):
        detector_entered = threading.Event()
        release_detector = threading.Event()
        publication_gap_reached = threading.Event()
        release_publication = threading.Event()
        cancel_started = threading.Event()
        cancel_entered_state_lock = threading.Event()
        cancel_finished = threading.Event()
        stop_states_at_sink = []

        class Detector:
            provider = "CPUExecutionProvider"
            input_size = 160

        def detector_factory(_path):
            detector_entered.set()
            release_detector.wait(1.0)
            return Detector()

        validator = ModelValidator(
            lambda _event: stop_states_at_sink.append(stop_event.is_set()),
            detector_factory=detector_factory,
        )
        self.addCleanup(validator.close)
        choice = ModelChoice(Path("chosen.onnx"), "chosen.onnx", False)
        self.assertTrue(validator.start(choice, 1))
        self.assertTrue(detector_entered.wait(1.0))
        stop_event = validator._active[1]

        class PublicationGapLock:
            def __init__(self):
                self._lock = threading.Lock()

            def __enter__(self):
                self._lock.acquire()
                if threading.current_thread().name == "ValidationCancel":
                    cancel_entered_state_lock.set()
                return self

            def __exit__(self, *_args):
                self._lock.release()
                if (
                    threading.current_thread().name == "ModelValidation-1"
                    and not publication_gap_reached.is_set()
                ):
                    publication_gap_reached.set()
                    release_publication.wait(5.0)

        validator._lock = PublicationGapLock()
        release_detector.set()
        self.assertTrue(publication_gap_reached.wait(1.0))

        def cancel():
            cancel_started.set()
            validator.cancel()
            cancel_finished.set()

        thread = threading.Thread(target=cancel, name="ValidationCancel")
        thread.start()
        self.assertTrue(cancel_started.wait(1.0))
        state_transition_entered = cancel_entered_state_lock.wait(1.0)
        release_publication.set()
        self.assertTrue(cancel_finished.wait(1.0))
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        self.assertFalse(state_transition_entered)
        self.assertEqual(stop_states_at_sink, [False])

    def test_failure_event_hides_exception_text_but_log_keeps_diagnostics(self):
        choice = ModelChoice(Path("secret.onnx"), "secret.onnx", False)
        events = []
        ready = threading.Event()

        def fail(_path):
            raise RuntimeError("sensitive absolute path detail")

        validator = ModelValidator(
            lambda event: (events.append(event), ready.set()),
            detector_factory=fail,
        )
        self.addCleanup(validator.close)
        with self.assertLogs("jitter_app.ai.model_selection", level="ERROR") as logs:
            self.assertTrue(validator.start(choice, 9))
            self.assertTrue(ready.wait(1.0))
        self.assertEqual(
            events,
            [
                ModelValidationEvent(
                    "error",
                    9,
                    choice,
                    "RuntimeError",
                    "AI model validation failed",
                )
            ],
        )
        self.assertNotIn("sensitive absolute path detail", repr(events))
        self.assertIn("sensitive absolute path detail", "\n".join(logs.output))

    def test_contract_failure_event_has_safe_actionable_message(self):
        events = []
        finished = threading.Event()

        def fail(_path):
            raise ModelContractError(
                "AI model input must use a 160, 320, or 640 square input"
            )

        choice = ModelChoice(Path("secret.onnx"), "secret.onnx", False)
        validator = ModelValidator(
            lambda event: (events.append(event), finished.set()),
            detector_factory=fail,
        )
        self.addCleanup(validator.close)
        with self.assertLogs("jitter_app.ai.model_selection", level="ERROR"):
            self.assertTrue(validator.start(choice, 8))
            self.assertTrue(finished.wait(1.0))
        self.assertEqual(events[0].error_type, "ModelContractError")
        self.assertEqual(
            events[0].safe_message,
            "AI model input must use a 160, 320, or 640 square input",
        )

    def test_dual_output_contract_failure_is_safe_and_actionable(self):
        events = []
        finished = threading.Event()
        message = (
            "AI model output must be output0 tensor(float) [1,300,6] "
            "or supported raw one-class [1,5,K]"
        )

        def fail(_path):
            raise ModelContractError(message)

        choice = ModelChoice(Path("private-model.onnx"), "private-model.onnx", False)
        validator = ModelValidator(
            lambda event: (events.append(event), finished.set()),
            detector_factory=fail,
        )
        self.addCleanup(validator.close)
        with self.assertLogs("jitter_app.ai.model_selection", level="ERROR"):
            self.assertTrue(validator.start(choice, 19))
            self.assertTrue(finished.wait(1.0))
        self.assertEqual(events[0].safe_message, message)
        self.assertNotIn(str(choice.path.resolve()), events[0].safe_message)

    def test_thread_start_failure_returns_false_without_duplicate_error_event(self):
        class FailingThread:
            def __init__(self, **_kwargs):
                pass

            def start(self):
                raise RuntimeError("scheduler detail")

        events = []
        validator = ModelValidator(events.append)
        self.addCleanup(validator.close)
        choice = ModelChoice(Path("chosen.onnx"), "chosen.onnx", False)
        with (
            mock.patch("jitter_app.ai.model_selection.threading.Thread", FailingThread),
            self.assertLogs("jitter_app.ai.model_selection", level="ERROR") as logs,
        ):
            self.assertFalse(validator.start(choice, 3))
        self.assertEqual(events, [])
        self.assertIn("scheduler detail", "\n".join(logs.output))
