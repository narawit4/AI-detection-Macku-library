import ast
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
    def test_ai_runtime_self_check_reports_verified_directml_json(self):
        output = io.StringIO()
        model = ROOT / "models" / "all_games_320.onnx"

        result = main.run_ai_runtime_self_check(
            model_path=model,
            detector_factory=lambda _path: SimpleNamespace(
                provider="DmlExecutionProvider"
            ),
            output=output,
        )

        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "expected_model_sha256":
                    "6B9157D6419F9DBC40D2DCECCC33A3387078C86F1C5872EDA544B174FF48499C",
                "model_path": str(model.resolve()),
                "model_sha256":
                    "6B9157D6419F9DBC40D2DCECCC33A3387078C86F1C5872EDA544B174FF48499C",
                "provider": "DmlExecutionProvider",
                "required_provider": "DmlExecutionProvider",
                "status": "ok",
            },
        )

    def test_ai_runtime_self_check_rejects_cpu_fallback(self):
        output = io.StringIO()

        result = main.run_ai_runtime_self_check(
            model_path=ROOT / "models" / "all_games_320.onnx",
            detector_factory=lambda _path: SimpleNamespace(
                provider="CPUExecutionProvider"
            ),
            output=output,
        )

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 1)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["provider"], "CPUExecutionProvider")
        self.assertEqual(payload["error"], "required_provider_unavailable")

    def test_ai_runtime_self_check_rejects_hash_and_contract_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            bad_model = Path(temporary) / "all_games_320.onnx"
            bad_model.write_bytes(b"not the approved model")
            detector_calls = []
            output = io.StringIO()

            result = main.run_ai_runtime_self_check(
                model_path=bad_model,
                detector_factory=lambda path: detector_calls.append(path),
                output=output,
            )

            self.assertEqual(result, 1)
            self.assertEqual(detector_calls, [])
            self.assertEqual(
                json.loads(output.getvalue())["error"],
                "model_hash_mismatch",
            )

        def reject_contract(_path):
            raise RuntimeError("model contract rejected")

        output = io.StringIO()
        result = main.run_ai_runtime_self_check(
            model_path=ROOT / "models" / "all_games_320.onnx",
            detector_factory=reject_contract,
            output=output,
        )
        self.assertEqual(result, 1)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["error"], "runtime_initialization_failed")
        self.assertNotIn("model contract rejected", output.getvalue())

    def test_self_check_import_failure_is_one_safe_json_object_without_traceback(self):
        secret = r"C:\private\native\onnxruntime.dll"
        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary)
            shutil.copy2(ROOT / "main.py", isolated / "main.py")
            (isolated / "ai_detection.py").write_text(
                f"raise ImportError({secret!r})\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(isolated / "main.py"),
                    "--ai-runtime-self-check",
                ],
                cwd=isolated,
                capture_output=True,
                text=True,
                timeout=10,
            )

        lines = completed.stdout.splitlines()
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(len(lines), 1, completed.stdout + completed.stderr)
        self.assertEqual(completed.stderr, "")
        payload = json.loads(lines[0])
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"], "runtime_initialization_failed")
        self.assertIsNone(payload["model_path"])
        self.assertNotIn(secret, completed.stdout + completed.stderr)
        self.assertNotIn("Traceback", completed.stdout + completed.stderr)

    def test_source_self_check_runs_without_gui_or_settings_modules(self):
        model = ROOT / "models" / "all_games_320.onnx"
        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary)
            shutil.copy2(ROOT / "main.py", isolated / "main.py")
            (isolated / "ai_detection.py").write_text(
                "from pathlib import Path\n"
                "class OnnxDetector:\n"
                "    def __init__(self, _path):\n"
                "        self.provider = 'DmlExecutionProvider'\n"
                "def model_resource_path():\n"
                f"    return Path({str(model)!r})\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(isolated / "main.py"),
                    "--ai-runtime-self-check",
                ],
                cwd=isolated,
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["provider"], "DmlExecutionProvider")
            self.assertFalse((isolated / "config.json").exists())
            self.assertFalse((isolated / "app.log").exists())

    @unittest.skipUnless(
        sys.platform == "win32",
        "DirectML source smoke requires the Windows build host",
    )
    def test_real_source_self_check_loads_bundled_model_with_directml(self):
        with tempfile.TemporaryDirectory() as temporary:
            working_directory = Path(temporary)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "main.py"),
                    "--ai-runtime-self-check",
                ],
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertFalse((working_directory / "config.json").exists())
            self.assertFalse((working_directory / "app.log").exists())

        self.assertEqual(
            completed.returncode, 0, completed.stdout + completed.stderr
        )
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["provider"], "DmlExecutionProvider")
        self.assertEqual(
            payload["model_path"],
            str((ROOT / "models" / "all_games_320.onnx").resolve()),
        )

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
                "pygame-ce": "==2.5.6",
                "onnxruntime-directml": "==1.24.4",
                "dxcam": "==0.3.0",
                "numpy": "==2.5.2",
                "pyserial": "==3.5",
                "comtypes": "==1.4.16",
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

    def test_distribution_cli_help_and_review_are_safe_and_review_uses_build_inputs(self):
        outputs = (ROOT / "build-output" / "Jitter.exe", ROOT / "build-output" / "build.log")
        before = tuple(
            (path.exists(), path.stat().st_mtime_ns if path.exists() else None)
            for path in outputs
        )
        completed = subprocess.run(
            [sys.executable, str(ROOT / "distribution_metadata.py"), "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("build-output\\jitter.exe", completed.stdout.lower())
        reviewed = subprocess.run(
            [sys.executable, str(ROOT / "distribution_metadata.py"), "--review-json"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(reviewed.returncode, 0, reviewed.stdout + reviewed.stderr)
        payload = json.loads(reviewed.stdout)
        expected_compile_targets = {
            "main.py", "ui.py", "motion.py", "ai_targeting.py", "ai_detection.py",
            "ai_capture.py", "ai_service.py", "makcu_service.py", "hotkeys.py",
            "settings.py", "liquid_widgets.py", "distribution_metadata.py",
        }
        expected_runtime_imports = {
            "makcu", "serial", "onnxruntime", "dxcam", "comtypes", "numpy",
        }
        expected_data_options = {
            "--include-data-dir=models=models",
            "--include-data-dir=licenses=licenses",
        }
        sound_is_present = (
            (ROOT / "sound_service.py").is_file() and (ROOT / "sound").is_dir()
        )
        if sound_is_present:
            expected_compile_targets.add("sound_service.py")
            expected_runtime_imports.add("pygame")
            expected_data_options.add("--include-data-dir=sound=sound")
        self.assertEqual(set(payload["compile_targets"]), expected_compile_targets)
        self.assertEqual(set(payload["runtime_imports"]), expected_runtime_imports)
        self.assertEqual(set(payload["nuitka_data_options"]), expected_data_options)
        self.assertEqual(
            set(payload["release_materials"]),
            {"LICENSE", "THIRD_PARTY_NOTICES.md", "licenses"},
        )
        self.assertEqual(
            {item["import_root"]: item["distribution"] for item in payload["runtime_inventory"]},
            {
                "makcu": "makcu", "serial": "pyserial",
                "onnxruntime": "onnxruntime-directml", "dxcam": "dxcam",
                "comtypes": "comtypes", "numpy": "numpy",
                **({"pygame": "pygame-ce"} if sound_is_present else {}),
            },
        )
        after = tuple(
            (path.exists(), path.stat().st_mtime_ns if path.exists() else None)
            for path in outputs
        )
        self.assertEqual(after, before)

    def test_distribution_cli_rejects_non_exact_vectors_without_shell_side_effects(self):
        invalid_vectors = (
            (),
            ("",),
            ("--not-a-build-mode",),
            ("-h",),
            ("--describe-build",),
            ("--help!",),
            ("--review-json!",),
            ('foo"&echo INJECTED',),
            ("foo&echo INJECTED",),
            ("foo|echo INJECTED",),
            ("100%",),
            ("!JITTER_LAUNCHER_TEST!",),
            ("", "--review-json"),
            ("--build", ""),
            ("--build", "extra"),
            ("--help", "extra"),
            ("--help", ""),
            ("--help", "--help"),
            ("--review-json", "extra"),
            ("--review-json", ""),
            ("--confirm-build", ""),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "injected.txt"
            redirect = f"foo>{marker}"
            invalid_vectors = (
                *invalid_vectors,
                (redirect,),
                (str(ROOT / "distribution_metadata.py"),),
            )
            for arguments in invalid_vectors:
                with self.subTest(arguments=arguments):
                    completed = subprocess.run(
                        [sys.executable, str(ROOT / "distribution_metadata.py"), *arguments],
                        cwd=root, capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(
                        completed.returncode, 2, completed.stdout + completed.stderr
                    )
                    self.assertNotIn(
                        "injected", (completed.stdout + completed.stderr).lower()
                    )
                    self.assertFalse(marker.exists())

    def test_gen_contains_no_user_argument_expansion_and_dispatches_only_confirmation(self):
        script_text = (ROOT / "gen.bat").read_text(encoding="utf-8").casefold()
        for forbidden in ("%1", "%~1", "%2", "%~2", "%*", "%cmdcmdline%"):
            self.assertNotIn(forbidden, script_text)
        self.assertIn(
            "python distribution_metadata.py --confirm-build", script_text
        )

    def test_gen_dispatches_fixed_confirmation_from_nested_interactive_and_powershell(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "gen.bat"
            script.write_bytes((ROOT / "gen.bat").read_bytes())
            marker = root / "helper-invocations.txt"
            (root / "python.cmd").write_text(
                "@echo off\r\n"
                f'echo %*>>\"{marker}\"\r\n'
                "exit /b 73\r\n",
                encoding="utf-8",
            )

            nested = subprocess.run(
                ["cmd.exe", "/d", "/c", "cmd.exe", "/d", "/c", str(script)],
                cwd=root, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(nested.returncode, 73, nested.stdout + nested.stderr)

            interactive = subprocess.run(
                ["cmd.exe", "/d", "/q"],
                cwd=root,
                input=f'"{script}"\r\n@echo JITTER_EXIT=%errorlevel%\r\n@exit\r\n',
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertIn("JITTER_EXIT=73", interactive.stdout)

            powershell = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                    "& '.\\gen.bat'; exit $LASTEXITCODE",
                ],
                cwd=root, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(
                powershell.returncode, 73, powershell.stdout + powershell.stderr
            )
            self.assertEqual(
                marker.read_text(encoding="utf-8").splitlines(),
                [
                    "distribution_metadata.py --confirm-build",
                    "distribution_metadata.py --confirm-build",
                    "distribution_metadata.py --confirm-build",
                ],
            )
            self.assertFalse((root / "build-output").exists())

    def test_gen_eof_and_negative_confirmation_propagate_cancellation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "gen.bat"
            script.write_bytes((ROOT / "gen.bat").read_bytes())
            (root / "distribution_metadata.py").write_bytes(
                (ROOT / "distribution_metadata.py").read_bytes()
            )

            eof = subprocess.run(
                ["cmd.exe", "/d", "/c", str(script)],
                cwd=root,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=10,
            )
            negative = subprocess.run(
                ["cmd.exe", "/d", "/c", str(script)],
                cwd=root,
                input="NO\n",
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertEqual(eof.returncode, 2, eof.stdout + eof.stderr)
            self.assertEqual(
                negative.returncode, 2, negative.stdout + negative.stderr
            )
            self.assertFalse((root / "build-output").exists())

    def test_source_tree_does_not_import_or_define_removed_stacks(self):
        forbidden_imports = {
            "ai_tracker", "ai_training", "torch", "ultralytics", "cv2",
            "mss", "customtkinter", "pystray", "pil",
        }
        imported_roots = set()
        source_paths = [
            path for path in ROOT.rglob("*.py")
            if not ({"tests", ".superpowers", "build-output", "__pycache__"}
                    & set(path.relative_to(ROOT).parts))
        ]
        for path in source_paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(alias.name.split(".", 1)[0].lower() for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.split(".", 1)[0].lower())
        self.assertTrue(imported_roots.isdisjoint(forbidden_imports), imported_roots & forbidden_imports)

        prohibited_features = {
            "training", "profile", "profiles", "overlay", "overlays", "tray",
            "ai_tracker",
        }
        source_tokens = {
            token
            for path in source_paths
            for part in (*path.relative_to(ROOT).parts, path.stem)
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
