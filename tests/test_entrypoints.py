import ast
import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from ai_detection import model_resource_path


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

    def test_runtime_requirements_pin_the_supported_stack(self):
        requirements = {}
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            name, separator, version = line.partition("==")
            self.assertEqual(separator, "==", f"runtime dependency is not pinned: {line}")
            self.assertTrue(name and version, f"invalid runtime dependency: {line}")
            normalized_name = re.sub(r"[-_.]+", "-", name).lower()
            self.assertNotIn(normalized_name, requirements, f"duplicate dependency: {line}")
            requirements[normalized_name] = f"=={version}"
        self.assertEqual(
            requirements,
            {
                "makcu": "==2.3.1",
                "onnxruntime-directml": "==1.24.4",
                "dxcam": "==0.3.0",
                "numpy": "==2.5.2",
            },
        )
        for forbidden in (
            "torch", "ultralytics", "opencv-python", "mss",
            "customtkinter", "pillow", "pystray",
        ):
            self.assertNotIn(forbidden, requirements)

    def test_approved_model_resource_exists_and_hash_matches(self):
        model = model_resource_path()
        self.assertEqual(model, ROOT / "models" / "all_games_320.onnx")
        self.assertTrue(model.is_file())
        self.assertEqual(
            hashlib.sha256(model.read_bytes()).hexdigest().upper(),
            "6B9157D6419F9DBC40D2DCECCC33A3387078C86F1C5872EDA544B174FF48499C",
        )

    def test_source_distribution_includes_licensing_artifacts(self):
        license_path = ROOT / "LICENSE"
        self.assertTrue(license_path.is_file())
        self.assertEqual(
            hashlib.sha256(license_path.read_bytes()).hexdigest().upper(),
            "0D96A4FF68AD6D4B6F1F30F713B18D5184912BA8DD389F86AA7710DB079ABCB0",
        )
        self.assertTrue((ROOT / "THIRD_PARTY_NOTICES.md").is_file())
        self.assertTrue((ROOT / "licenses" / "manifest.json").is_file())

    def test_gen_help_and_review_are_safe_and_review_uses_build_inputs(self):
        outputs = (ROOT / "build-output" / "Jitter.exe", ROOT / "build-output" / "build.log")
        before = tuple(
            (path.exists(), path.stat().st_mtime_ns if path.exists() else None)
            for path in outputs
        )
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", str(ROOT / "gen.bat"), "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("build-output\\jitter.exe", completed.stdout.lower())
        reviewed = subprocess.run(
            ["cmd.exe", "/d", "/c", str(ROOT / "gen.bat"), "--review-json"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(reviewed.returncode, 0, reviewed.stdout + reviewed.stderr)
        payload = json.loads(reviewed.stdout)
        self.assertTrue(
            {"ai_capture.py", "ai_detection.py", "ai_targeting.py", "ai_service.py"}
            <= set(payload["compile_targets"])
        )
        self.assertIn("--include-data-dir=models=models", payload["nuitka_data_options"])
        self.assertIn("--include-data-dir=licenses=licenses", payload["nuitka_data_options"])
        self.assertEqual(
            set(payload["release_materials"]),
            {"LICENSE", "THIRD_PARTY_NOTICES.md", "licenses"},
        )
        after = tuple(
            (path.exists(), path.stat().st_mtime_ns if path.exists() else None)
            for path in outputs
        )
        self.assertEqual(after, before)

    def test_gen_rejects_unknown_options_without_building(self):
        output = ROOT / "build-output" / "Jitter.exe"
        before = (output.exists(), output.stat().st_mtime_ns if output.exists() else None)
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", str(ROOT / "gen.bat"), "--not-a-build-mode"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("unknown option", completed.stderr.lower())
        after = (output.exists(), output.stat().st_mtime_ns if output.exists() else None)
        self.assertEqual(after, before)

    def test_source_tree_does_not_import_or_define_removed_stacks(self):
        forbidden_imports = {
            "ai_tracker", "ai_training", "torch", "ultralytics", "cv2",
            "mss", "customtkinter", "pystray", "pil",
        }
        imported_roots = set()
        for path in ROOT.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(alias.name.split(".", 1)[0].lower() for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.split(".", 1)[0].lower())
        self.assertTrue(imported_roots.isdisjoint(forbidden_imports), imported_roots & forbidden_imports)

        prohibited_features = {"training", "profile", "overlay", "tray", "ai_tracker"}
        source_tokens = {
            token
            for path in ROOT.glob("*.py")
            for part in (*path.parts, path.stem)
            for token in re.split(r"[-_.]+", part.lower())
        }
        self.assertTrue(
            source_tokens.isdisjoint(prohibited_features),
            source_tokens & prohibited_features,
        )

    def test_readme_links_release_compliance_materials(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        links = set(re.findall(r"\[[^]]+\]\(([^)]+)\)", readme))
        self.assertIn("THIRD_PARTY_NOTICES.md", links)
        self.assertIn("licenses/README.md", links)
