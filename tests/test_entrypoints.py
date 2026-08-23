import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import main


ROOT = Path(__file__).parents[1]


class FakeKernel32:
    def __init__(self, last_error, handle=123):
        self.last_error = last_error
        self.handle = handle
        self.closed = []

    def CreateMutexW(self, _security, _owner, _name):
        return self.handle

    def GetLastError(self):
        return self.last_error

    def CloseHandle(self, handle):
        self.closed.append(handle)


class EntryPointTests(unittest.TestCase):
    def test_second_instance_returns_no_handle_and_closes_duplicate(self):
        kernel32 = FakeKernel32(last_error=183)
        self.assertIsNone(main.ensure_single_instance(kernel32))
        self.assertEqual(kernel32.closed, [123])

    def test_first_instance_keeps_mutex_handle(self):
        kernel32 = FakeKernel32(last_error=0)
        self.assertEqual(main.ensure_single_instance(kernel32), 123)
        self.assertEqual(kernel32.closed, [])

    def test_mutex_creation_failure_is_distinct_from_duplicate(self):
        kernel32 = FakeKernel32(last_error=5, handle=0)
        with self.assertRaises(main.MutexCreationError) as raised:
            main.ensure_single_instance(kernel32)
        self.assertEqual(raised.exception.error_code, 5)
        self.assertEqual(kernel32.closed, [])

    def test_main_reports_mutex_creation_failure_as_startup_error(self):
        with patch.object(main, "runtime_base_dir", return_value=ROOT), patch.object(
            main, "configure_logging"
        ), patch.object(
            main,
            "ensure_single_instance",
            side_effect=main.MutexCreationError(5),
        ), patch.object(main, "_show_startup_error") as show_error, patch.object(
            main, "JitterApp"
        ) as app_factory, patch.object(main.logging, "error") as _log_error:
            main.main()

        show_error.assert_called_once()
        self.assertIn("mutex", show_error.call_args.args[0].lower())
        app_factory.assert_not_called()

    def test_runtime_requirements_exclude_removed_feature_stacks(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertEqual(requirements.strip(), "makcu==2.3.1")
        for forbidden in ("torch", "onnx", "pillow", "pystray", "ultralytics"):
            self.assertNotIn(forbidden, requirements)

    def test_gen_help_is_safe_and_documents_explicit_build(self):
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", str(ROOT / "gen.bat"), "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("build-output\\jitter.exe", completed.stdout.lower())

    def test_source_tree_does_not_import_removed_stacks(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in ROOT.glob("*.py")
        ).lower()
        for forbidden in ("ai_tracker", "ai_training", "onnxruntime", "torch", "pystray"):
            self.assertNotIn(forbidden, source)
