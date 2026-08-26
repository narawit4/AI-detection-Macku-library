import hashlib
import io
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from distribution_metadata import (
    build_plan,
    confirm_build,
    copy_release_materials,
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
    def test_review_serializes_the_exact_command_plan_executed_by_build_mode(self):
        plan = build_plan(ROOT)
        payload = review_payload(root=ROOT)
        self.assertEqual(payload, plan.to_payload())

        expected_data_options = {
            "--include-data-dir=models=models",
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
                "main.py", "ui.py", "motion.py", "ai_targeting.py",
                "ai_detection.py", "ai_capture.py", "ai_service.py",
                "makcu_service.py", "hotkeys.py", "settings.py",
                "liquid_widgets.py", "distribution_metadata.py",
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

        calls = []
        copies = []

        def runner(argv, **kwargs):
            calls.append(tuple(argv))
            return SimpleNamespace(returncode=0)

        def release_copier(output_dir, *, root, materials):
            copies.append((Path(output_dir), root, tuple(materials)))
            return ()

        with tempfile.TemporaryDirectory() as temporary:
            executable_plan = replace(
                plan,
                output_dir=Path(temporary),
                build_log=Path(temporary) / "build.log",
            )
            execute_build_plan(
                executable_plan,
                runner=runner,
                release_copier=release_copier,
            )

        self.assertEqual(
            calls,
            [
                executable_plan.install_argv,
                executable_plan.compile_argv,
                executable_plan.test_argv,
                executable_plan.runtime_import_argv,
                executable_plan.nuitka_argv,
            ],
        )
        self.assertEqual(
            copies,
            [(executable_plan.output_dir, executable_plan.root,
              executable_plan.release_materials)],
        )

    def test_new_third_party_source_import_requires_pin_license_and_root_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(ROOT / "licenses", root / "licenses")
            shutil.copy2(ROOT / "LICENSE", root / "LICENSE")
            shutil.copy2(
                ROOT / "THIRD_PARTY_NOTICES.md", root / "THIRD_PARTY_NOTICES.md"
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
