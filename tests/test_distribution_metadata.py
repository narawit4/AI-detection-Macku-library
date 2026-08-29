import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path, PurePosixPath

from distribution_metadata import (
    build_plan,
    confirm_build,
    copy_release_materials,
    discover_application_sources,
    execute_build_plan,
    parse_pinned_requirements,
    review_payload,
    validate_license_manifest,
    validate_runtime_inventory,
)


ROOT = Path(__file__).parents[1]


class RequirementMetadataTests(unittest.TestCase):
    def test_requirement_names_are_normalized_and_duplicate_names_are_rejected(self):
        parsed = parse_pinned_requirements(
            "ONNX_Runtime.DirectML==1.24.4\ndxcam==0.3.0\n"
        )
        self.assertEqual(
            [(item.name, item.version) for item in parsed],
            [("onnx-runtime-directml", "1.24.4"), ("dxcam", "0.3.0")],
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_pinned_requirements("DXCam==0.3.0\ndxcam==0.3.0\n")

    def test_repository_manifest_matches_requirements_and_artifact_hashes(self):
        manifest = json.loads(
            (ROOT / "licenses" / "manifest.json").read_text(encoding="utf-8")
        )
        requirements = parse_pinned_requirements(
            (ROOT / "requirements.txt").read_text(encoding="utf-8")
        )
        records = validate_license_manifest(manifest, ROOT, requirements)
        record_versions = {record.name: record.version for record in records}
        self.assertEqual(
            set(record_versions),
            {
                "makcu", "pygame-ce", "numpy", "onnxruntime-directml", "dxcam",
                "pyserial", "comtypes",
            },
        )
        for item in requirements:
            self.assertEqual(record_versions[item.name], item.version)
        copyleft = {record.name: record for record in records if record.source_required}
        self.assertEqual(set(copyleft), {"makcu", "pygame-ce"})
        for record in copyleft.values():
            self.assertTrue(record.source_archive.is_file())

    def test_manifest_records_reviewed_metadata_and_core_artifact_hashes(self):
        manifest = json.loads(
            (ROOT / "licenses" / "manifest.json").read_text(encoding="utf-8")
        )
        for package in manifest["packages"]:
            self.assertTrue(
                {
                    "name", "version", "license", "copyright", "homepage",
                    "metadata_provenance", "license_files", "source",
                }
                <= set(package)
            )

        expected_hashes = {
            "licenses/makcu-2.3.1/LICENSE":
                "230184F60BAE2FEAF244F10A8BAC053C8FF33A183BCC365B4D8B876D2B7F4809",
            "licenses/pygame-ce-2.5.6/LGPL-2.1.txt":
                "885A03F54B157961236F46843E79972ABFCD6890B6CBB368BC7ECA328FF95A12",
            "licenses/numpy-2.5.2/LICENSE.txt":
                "A804DFF0EAD9FADC5293456410BCBFC32BF024BE9C4513459663FB7B442D2341",
            "licenses/onnxruntime-directml-1.24.4/LICENSE":
                "C250D6278F0B47A6439FB7592B08B58A55EB9F535AA49A1DB63211C3F982B674",
            "licenses/onnxruntime-directml-1.24.4/ThirdPartyNotices.txt":
                "FB0AF774B4D7CFFC5B9D046F2AAEADE2F37DF2F80ABF8033C95DFFFCC77A8866",
            "licenses/dxcam-0.3.0/LICENSE":
                "4FE6BAEE928B96D2CF0F6A238275ACFD86182CDAEC6E8146654F34CF08C1C9B3",
            "licenses/pyserial-3.5/LICENSE.txt":
                "F91CB9813DE6A5B142B8F7F2DEDE630B5134160AEDAEAF55F4D6A7E2593CA3F3",
            "licenses/comtypes-1.4.16/LICENSE.txt":
                "3B1767F010980B46926B23BF0AFCE5D72F3359EE5E2B27BACA71B9B4209AB383",
            "licenses/sources/makcu-2.3.1.tar.gz":
                "DA94880094DA55E83FDD8E2BB7CD16D4621E46A92839E36BFB8DF3ABFCECE7F5",
            "licenses/sources/pygame_ce-2.5.6.tar.gz":
                "D3D019309D1E76FD19978B01753E8576BD76C66411AC7A4885785F95E68DC261",
        }
        for relative, expected_hash in expected_hashes.items():
            actual_hash = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest().upper()
            self.assertEqual(actual_hash, expected_hash, relative)

    def test_runtime_inventory_requires_every_import_root_to_be_pinned_and_licensed(self):
        requirements = parse_pinned_requirements(
            "makcu==2.3.1\npyserial==3.5\ndxcam==0.3.0\ncomtypes==1.4.16\n"
        )
        packages = {
            "makcu": "2.3.1",
            "pyserial": "3.5",
            "dxcam": "0.3.0",
            "comtypes": "1.4.16",
        }
        inventory = [
            {"import_root": "makcu", "distribution": "makcu", "required_by": "Jitter"},
            {"import_root": "serial", "distribution": "pyserial", "required_by": "makcu==2.3.1"},
            {"import_root": "dxcam", "distribution": "dxcam", "required_by": "Jitter"},
            {"import_root": "comtypes", "distribution": "comtypes", "required_by": "dxcam==0.3.0"},
        ]

        validated = validate_runtime_inventory(
            inventory, requirements, packages, {"makcu", "serial", "dxcam", "comtypes"}
        )
        self.assertEqual(
            {(item.import_root, item.distribution) for item in validated},
            {("makcu", "makcu"), ("serial", "pyserial"),
             ("dxcam", "dxcam"), ("comtypes", "comtypes")},
        )

        with self.assertRaisesRegex(ValueError, "missing"):
            validate_runtime_inventory(
                inventory, requirements, packages, {"makcu", "serial", "missing"}
            )


class ReleaseMaterialTests(unittest.TestCase):
    def test_copy_release_materials_places_files_and_license_tree_beside_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            output = Path(temporary) / "release"
            (root / "licenses" / "package").mkdir(parents=True)
            (root / "LICENSE").write_text("project license", encoding="utf-8")
            (root / "THIRD_PARTY_NOTICES.md").write_text("notices", encoding="utf-8")
            (root / "licenses" / "package" / "LICENSE").write_text(
                "package license", encoding="utf-8"
            )

            copied = copy_release_materials(
                output,
                root=root,
                materials=("LICENSE", "THIRD_PARTY_NOTICES.md", "licenses"),
            )

            self.assertEqual(
                copied,
                (output / "LICENSE", output / "THIRD_PARTY_NOTICES.md", output / "licenses"),
            )
            self.assertEqual(
                (output / "licenses" / "package" / "LICENSE").read_text(encoding="utf-8"),
                "package license",
            )


class BuildPlanTests(unittest.TestCase):
    def test_source_discovery_ignores_nested_git_worktrees(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live_source = root / "main.py"
            nested_source = root / ".worktrees" / "feature" / "main.py"
            nested_source.parent.mkdir(parents=True)
            live_source.write_text("pass\n", encoding="utf-8")
            nested_source.write_text("pass\n", encoding="utf-8")

            self.assertEqual(discover_application_sources(root), (live_source,))

    def test_user_package_config_targets_exact_installed_directml_dll(self):
        from nuitka.Tracing import general
        from nuitka.utils.FileOperations import listDllFilesFromDirectory
        from nuitka.utils.Yaml import PackageConfigYaml

        config_path = ROOT / "nuitka-package.config.yml"
        parsed = PackageConfigYaml(
            logger=general,
            name=str(config_path),
            file_data=config_path.read_bytes(),
            assume_yes_for_downloads=False,
            check_checksums=False,
        )
        dll_config = parsed.get("ai_detection", section="dlls")
        self.assertEqual(len(dll_config), 1)
        rule = dll_config[0]
        source = rule["from_filenames"]
        self.assertEqual(source["relative_to"], "onnxruntime.capi")
        self.assertEqual(source["relative_path"], ".")
        self.assertEqual(source["prefixes"], ["DirectML"])
        self.assertEqual(source["suffixes"], ["dll"])
        self.assertEqual(rule["dest_path"], "onnxruntime/capi")
        self.assertEqual(rule["when"], "win32")

        capi_spec = importlib.util.find_spec(source["relative_to"])
        self.assertIsNotNone(capi_spec)
        self.assertIsNotNone(capi_spec.submodule_search_locations)
        capi_dir = Path(next(iter(capi_spec.submodule_search_locations)))
        matches = [
            Path(path)
            for prefix in source["prefixes"]
            for path, _name in listDllFilesFromDirectory(
                str(capi_dir / source["relative_path"]),
                prefix=prefix,
                suffixes=tuple(source["suffixes"]),
            )
        ]
        installed_dll = capi_dir / "DirectML.dll"
        self.assertEqual(matches, [installed_dll])
        self.assertEqual(
            hashlib.sha256(installed_dll.read_bytes()).hexdigest().upper(),
            "B73972115320E906A49602F2027A3266622881B0D325BA685E0F165A9482A8D7",
        )
        self.assertEqual(
            PurePosixPath(rule["dest_path"]) / installed_dll.name,
            PurePosixPath("onnxruntime/capi/DirectML.dll"),
        )

    def test_review_serializes_the_exact_command_plan_executed_by_build_mode(self):
        plan = build_plan(ROOT)
        payload = review_payload(root=ROOT)
        self.assertEqual(payload, plan.to_payload())

        expected_package_configuration = {
            "path": "nuitka-package.config.yml",
            "config_sha256":
                "3B41E39B66EBB8E28BD728933F03C4B5B8DA21C3C87C1433742D368D07140452",
            "module": "ai_detection",
            "source": "onnxruntime/capi/DirectML.dll",
            "destination": "onnxruntime/capi/DirectML.dll",
            "sha256":
                "B73972115320E906A49602F2027A3266622881B0D325BA685E0F165A9482A8D7",
        }
        self.assertEqual(
            payload["nuitka_package_configuration"],
            expected_package_configuration,
        )
        package_option = (
            "--user-package-configuration-file=nuitka-package.config.yml"
        )
        self.assertIn(package_option, plan.nuitka_argv)
        self.assertIn("--windows-console-mode=attach", plan.nuitka_argv)
        self.assertNotIn("--windows-console-mode=disable", plan.nuitka_argv)
        self.assertEqual(
            tuple(payload["commands"]["packaged_self_check"]),
            plan.packaged_self_check_argv,
        )
        self.assertEqual(
            plan.packaged_self_check_argv[-1],
            "--ai-runtime-self-check",
        )

        expected_data_options = {
            "--include-data-files="
            "models/all_games_320.onnx=models/all_games_320.onnx",
            "--include-data-dir=licenses=licenses",
        }
        if (ROOT / "sound_service.py").is_file() and (ROOT / "sound").is_dir():
            expected_data_options.add("--include-data-dir=sound=sound")
        self.assertEqual(set(plan.nuitka_data_options), expected_data_options)
        for option in expected_data_options:
            self.assertIn(option, plan.nuitka_argv)
        self.assertEqual(
            set(plan.compile_targets),
            {
                "main.py", "ui.py", "motion.py", "combined_motion.py",
                "ai_targeting.py", "ai_tracking.py", "ai_detection.py", "ai_yolo.py",
                "ai_model_selection.py",
                "ai_capture.py", "ai_zoom.py", "image_resize.py", "ai_service.py",
                "display_timing.py", "overlay.py", "makcu_service.py",
                "hotkeys.py", "settings.py", "liquid_widgets.py",
                "distribution_metadata.py",
                "jitter_app/__init__.py", "jitter_app/resources.py",
                "jitter_app/ai/__init__.py", "jitter_app/motion/__init__.py",
                "jitter_app/device/__init__.py",
                "jitter_app/presentation/__init__.py",
                "jitter_app/config/__init__.py",
                *({"sound_service.py"} if (ROOT / "sound_service.py").is_file() else set()),
            },
        )
        self.assertEqual(
            tuple(payload["commands"]["nuitka"]), plan.nuitka_argv
        )
        self.assertEqual(
            plan.compile_argv[-len(plan.compile_targets):], plan.compile_targets
        )
        self.assertEqual(
            plan.runtime_import_argv[-1],
            "import " + ", ".join(plan.runtime_imports),
        )
        self.assertEqual(
            tuple(payload["commands"]["compile"]), plan.compile_argv
        )
        self.assertEqual(
            tuple(payload["commands"]["runtime_import"]),
            plan.runtime_import_argv,
        )

        events = []

        def runner(argv, **kwargs):
            self.assertEqual(kwargs["cwd"], executable_plan.root)
            self.assertIs(kwargs["check"], True)
            self.assertIn("env", kwargs)
            self.assertIsNone(kwargs["env"])
            events.append(("run", tuple(argv), kwargs))
            return subprocess.CompletedProcess(argv, 0)

        def release_copier(output_dir, *, root, materials):
            events.append(
                ("copy", (Path(output_dir), root, tuple(materials)))
            )
            return ()

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "release-output"
            executable_plan = replace(
                plan,
                output_dir=output_dir,
                build_log=output_dir / "build.log",
                nuitka_argv=tuple(
                    f"--output-dir={output_dir}"
                    if argument.startswith("--output-dir=")
                    else argument
                    for argument in plan.nuitka_argv
                ),
                packaged_self_check_argv=(
                    str(output_dir / "Jitter.exe"),
                    plan.packaged_self_check_argv[-1],
                ),
            )
            execute_build_plan(
                executable_plan,
                runner=runner,
                release_copier=release_copier,
            )

        self.assertEqual(
            [event[1] for event in events if event[0] == "run"],
            [
                executable_plan.install_argv,
                executable_plan.compile_argv,
                executable_plan.test_argv,
                executable_plan.runtime_import_argv,
                executable_plan.nuitka_argv,
                executable_plan.packaged_self_check_argv,
            ],
        )
        self.assertIn(
            f"--output-dir={executable_plan.output_dir}",
            executable_plan.nuitka_argv,
        )
        self.assertEqual(
            events[-1],
            (
                "copy",
                (
                    executable_plan.output_dir,
                    executable_plan.root,
                    executable_plan.release_materials,
                ),
            ),
        )
        for event in events[:4] + events[5:6]:
            self.assertEqual(
                set(event[2]),
                {"cwd", "check", "env"},
            )
        nuitka_kwargs = events[4][2]
        self.assertEqual(
            set(nuitka_kwargs),
            {"cwd", "check", "env", "stdout", "stderr"},
        )
        self.assertEqual(
            nuitka_kwargs["stdout"].name,
            str(executable_plan.build_log),
        )
        self.assertIs(nuitka_kwargs["stderr"], subprocess.STDOUT)

    def test_canonical_package_data_includes_only_the_bundled_model_file(self):
        plan = build_plan(ROOT)

        model_options = tuple(
            option for option in plan.nuitka_data_options
            if "models/" in option or "models=" in option
        )
        self.assertEqual(
            model_options,
            (
                "--include-data-files="
                "models/all_games_320.onnx=models/all_games_320.onnx",
            ),
        )
        self.assertNotIn("--include-data-dir=models=models", plan.nuitka_argv)

    def test_failed_packaged_self_check_prevents_release_copying(self):
        plan = build_plan(ROOT)
        copied = []

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "release-output"
            executable_plan = replace(
                plan,
                output_dir=output_dir,
                build_log=output_dir / "build.log",
                nuitka_argv=tuple(
                    f"--output-dir={output_dir}"
                    if argument.startswith("--output-dir=")
                    else argument
                    for argument in plan.nuitka_argv
                ),
                packaged_self_check_argv=(
                    str(output_dir / "Jitter.exe"),
                    plan.packaged_self_check_argv[-1],
                ),
            )

            calls = []

            def runner(argv, **kwargs):
                command = tuple(argv)
                calls.append((command, kwargs))
                returncode = int(
                    command == executable_plan.packaged_self_check_argv
                )
                completed = subprocess.CompletedProcess(argv, returncode)
                if returncode and kwargs.get("check"):
                    raise subprocess.CalledProcessError(returncode, argv)
                return completed

            unchecked = runner(
                executable_plan.packaged_self_check_argv,
                cwd=executable_plan.root,
                check=False,
                env=None,
            )
            self.assertEqual(unchecked.returncode, 1)
            calls.clear()
            with self.assertRaises(subprocess.CalledProcessError):
                execute_build_plan(
                    executable_plan,
                    runner=runner,
                    release_copier=lambda *_args, **_kwargs: copied.append(True),
                )

        self.assertEqual(copied, [])
        self.assertEqual(calls[-1][0], executable_plan.packaged_self_check_argv)
        self.assertTrue(all(call[1]["check"] for call in calls))
        self.assertTrue(all(call[1]["env"] is None for call in calls))

    def test_new_third_party_source_import_requires_pin_license_and_root_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(ROOT / "licenses", root / "licenses")
            shutil.copy2(ROOT / "LICENSE", root / "LICENSE")
            shutil.copy2(
                ROOT / "THIRD_PARTY_NOTICES.md", root / "THIRD_PARTY_NOTICES.md"
            )
            shutil.copy2(
                ROOT / "nuitka-package.config.yml",
                root / "nuitka-package.config.yml",
            )
            (root / "requirements.txt").write_text(
                "makcu==2.3.1\npyserial==3.5\n"
                "onnxruntime-directml==1.24.4\ndxcam==0.3.0\n"
                "comtypes==1.4.16\nnumpy==2.5.2\n",
                encoding="utf-8",
            )
            (root / "main.py").write_text(
                "import makcu\nimport onnxruntime\nimport dxcam\nimport numpy\n",
                encoding="utf-8",
            )
            (root / "models").mkdir()
            shutil.copy2(
                ROOT / "models" / "all_games_320.onnx",
                root / "models" / "all_games_320.onnx",
            )

            build_plan(root)
            (root / "feature").mkdir()
            (root / "feature" / "adapter.py").write_text(
                "import mystery_sdk\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "mystery_sdk"):
                build_plan(root)

    def test_package_configuration_drift_stops_plan_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(ROOT / "licenses", root / "licenses")
            shutil.copytree(ROOT / "models", root / "models")
            for name in (
                "LICENSE",
                "THIRD_PARTY_NOTICES.md",
                "requirements.txt",
                "main.py",
                "nuitka-package.config.yml",
            ):
                shutil.copy2(ROOT / name, root / name)

            config_path = root / "nuitka-package.config.yml"
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace(
                    "DirectML", "NotDirectML"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "package configuration"):
                build_plan(root)


class BuildConfirmationTests(unittest.TestCase):
    def test_eof_and_non_exact_confirmation_do_not_create_or_execute_a_plan(self):
        for response in (EOFError(), KeyboardInterrupt(), "", "build", "BUILD "):
            with self.subTest(response=response):
                calls = []

                def read_confirmation(_prompt):
                    if isinstance(response, BaseException):
                        raise response
                    return response

                result = confirm_build(
                    input_fn=read_confirmation,
                    output=io.StringIO(),
                    plan_factory=lambda: calls.append("plan"),
                    executor=lambda plan: calls.append(plan),
                )

                self.assertEqual(result, 2)
                self.assertEqual(calls, [])

    def test_exact_confirmation_executes_the_stubbed_plan(self):
        plan = object()
        executed = []

        result = confirm_build(
            input_fn=lambda _prompt: "BUILD",
            output=io.StringIO(),
            plan_factory=lambda: plan,
            executor=executed.append,
        )

        self.assertEqual(result, 0)
        self.assertEqual(executed, [plan])


if __name__ == "__main__":
    unittest.main()
