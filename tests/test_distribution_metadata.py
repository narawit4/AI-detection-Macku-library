import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from distribution_metadata import (
    copy_release_materials,
    parse_pinned_requirements,
    validate_license_manifest,
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
            {"makcu", "pygame-ce", "numpy", "onnxruntime-directml", "dxcam"},
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
            "licenses/sources/makcu-2.3.1.tar.gz":
                "DA94880094DA55E83FDD8E2BB7CD16D4621E46A92839E36BFB8DF3ABFCECE7F5",
            "licenses/sources/pygame_ce-2.5.6.tar.gz":
                "D3D019309D1E76FD19978B01753E8576BD76C66411AC7A4885785F95E68DC261",
        }
        for relative, expected_hash in expected_hashes.items():
            actual_hash = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest().upper()
            self.assertEqual(actual_hash, expected_hash, relative)


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


if __name__ == "__main__":
    unittest.main()
