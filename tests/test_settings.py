import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_targeting import AimSettings
from motion import MotionSettings
import settings as settings_module
from settings import AppConfig, ConfigStore, SCHEMA_VERSION, runtime_base_dir


class RuntimeBaseDirTests(unittest.TestCase):
    def test_source_mode_uses_module_directory_despite_packaging_inputs(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            settings_module.os.environ,
            {"NUITKA_ONEFILE_DIRECTORY": directory},
        ), patch.object(
            settings_module.sys,
            "argv",
            [str(Path(directory) / "Jitter.exe")],
        ):
            self.assertEqual(
                runtime_base_dir(),
                Path(settings_module.__file__).resolve().parent,
            )

    def test_nuitka_standalone_and_onefile_use_containing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            containing_dir = Path(directory) / "release"
            for onefile in (False, True):
                marker = SimpleNamespace(
                    containing_dir=str(containing_dir),
                    standalone=True,
                    onefile=onefile,
                )
                with self.subTest(onefile=onefile), patch.object(
                    settings_module,
                    "__compiled__",
                    marker,
                    create=True,
                ), patch.object(
                    settings_module.sys,
                    "argv",
                    [str(Path(directory) / "elsewhere" / "Jitter.exe")],
                ):
                    self.assertEqual(
                        runtime_base_dir(), containing_dir.resolve()
                    )

    def test_malformed_compiled_marker_falls_back_to_executable_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "release" / "Jitter.exe"
            for marker in (object(), SimpleNamespace(containing_dir=None)):
                with self.subTest(marker=marker), patch.object(
                    settings_module,
                    "__compiled__",
                    marker,
                    create=True,
                ), patch.object(
                    settings_module.sys,
                    "argv",
                    [str(executable)],
                ):
                    self.assertEqual(runtime_base_dir(), executable.resolve().parent)

    def test_malformed_compiled_paths_fall_back_safely_to_source_directory(self):
        marker = SimpleNamespace(containing_dir=object())
        with patch.object(
            settings_module,
            "__compiled__",
            marker,
            create=True,
        ), patch.object(settings_module.sys, "argv", [None]):
            self.assertEqual(
                runtime_base_dir(),
                Path(settings_module.__file__).resolve().parent,
            )


class ConfigStoreTests(unittest.TestCase):
    def test_missing_config_returns_safe_disabled_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            outcome = ConfigStore(Path(directory) / "config.json").load()
        self.assertEqual(outcome.config.motion, MotionSettings())
        self.assertEqual(outcome.config.trigger, "Left")
        self.assertEqual(outcome.config.modifier, "None")
        self.assertEqual(outcome.config.selected_preset, "Balanced")
        self.assertEqual(outcome.config.theme, "light")
        self.assertTrue(outcome.save_allowed)

    def test_schema_three_round_trip_saves_only_paired_pulse_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            store = ConfigStore(path)
            config = AppConfig(
                motion=MotionSettings(4.0, 45.0, "Instant"),
                selected_preset="Strong",
            )
            store.save(config)
            document = json.loads(path.read_text(encoding="utf-8"))
            restored = store.load().config
        self.assertEqual(document["schema_version"], 3)
        self.assertEqual(document["motion"], {
            "pulse_size_px": "4", "pulse_rate_hz": "45", "ramp_mode": "Instant",
        })
        self.assertNotIn("enabled", document)
        self.assertNotIn("moving", document)
        self.assertEqual(restored, config)

    def test_schema_two_migrates_to_jitter_and_safe_ai_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "schema_version": 2,
                "motion": {
                    "pulse_size_px": "4",
                    "pulse_rate_hz": "45",
                    "ramp_mode": "Instant",
                },
                "mode": "ai_aim",
                "ai": {"confidence": "0.9"},
            }), encoding="utf-8")
            outcome = ConfigStore(path).load()
        self.assertEqual(outcome.config.mode, "jitter")
        self.assertEqual(outcome.config.ai, AimSettings())
        self.assertEqual(
            outcome.config.motion,
            MotionSettings(4.0, 45.0, "Instant"),
        )

    def test_schema_three_round_trips_ai_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = AppConfig(
                mode="ai_aim",
                ai=AimSettings(0.5, 0.6, 0.7, 30),
            )
            store = ConfigStore(path)
            store.save(config)
            document = json.loads(path.read_text(encoding="utf-8"))
            restored = store.load().config
        self.assertEqual(document["mode"], "ai_aim")
        self.assertEqual(document["ai"], {
            "confidence": "0.5",
            "aim_strength": "0.6",
            "smoothing": "0.7",
            "max_step": "30",
        })
        self.assertEqual(restored, config)

    def test_schema_three_invalid_mode_uses_jitter_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "schema_version": 3,
                "mode": "unknown",
            }), encoding="utf-8")
            config = ConfigStore(path).load().config
        self.assertEqual(config.mode, "jitter")

    def test_schema_three_malformed_ai_settings_use_safe_defaults(self):
        cases = (
            None,
            ["not", "an", "object"],
            {"confidence": "bad", "aim_strength": None,
             "smoothing": "infinite", "max_step": "many"},
        )
        for raw_ai in cases:
            with self.subTest(ai=raw_ai), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps({
                    "schema_version": 3,
                    "ai": raw_ai,
                }), encoding="utf-8")
                config = ConfigStore(path).load().config
            self.assertEqual(config.ai, AimSettings())

    def test_schema_three_ai_settings_are_validated_and_clamped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "schema_version": 3,
                "ai": {
                    "confidence": -1,
                    "aim_strength": 10,
                    "smoothing": "9",
                    "max_step": 999,
                },
            }), encoding="utf-8")
            config = ConfigStore(path).load().config
        self.assertEqual(config.ai, AimSettings(0.05, 2.0, 0.95, 127))

    def test_schema_three_boolean_ai_values_use_field_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "schema_version": 3,
                "ai": {
                    "confidence": True,
                    "aim_strength": "1.25",
                    "smoothing": False,
                    "max_step": "30",
                },
            }), encoding="utf-8")
            config = ConfigStore(path).load().config
        self.assertEqual(config.ai, AimSettings(0.35, 1.25, 0.65, 30))

    def test_invalid_theme_uses_safe_light_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "schema_version": SCHEMA_VERSION,
                "theme": "midnight",
            }), encoding="utf-8")
            config = ConfigStore(path).load().config
        self.assertEqual(config.theme, "light")

    def test_second_save_keeps_previous_document_as_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            store = ConfigStore(path)
            first = AppConfig(selected_preset="Soft")
            second = AppConfig(selected_preset="Strong")
            store.save(first)
            store.save(second)
            backup = json.loads((Path(str(path) + ".bak")).read_text(encoding="utf-8"))
        self.assertEqual(backup["selected_preset"], "Soft")

    def test_future_schema_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            original = {"schema_version": SCHEMA_VERSION + 1, "future": True}
            path.write_text(json.dumps(original), encoding="utf-8")
            store = ConfigStore(path)
            outcome = store.load()
            self.assertFalse(outcome.save_allowed)
            with self.assertRaises(PermissionError):
                store.save(outcome.config)
            restored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(restored, original)

    def test_fractional_newer_schema_is_not_truncated_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            original_text = json.dumps({
                "schema_version": 2.5,
                "future": {"keep": True},
            })
            path.write_text(original_text, encoding="utf-8")
            store = ConfigStore(path)
            outcome = store.load()
            self.assertFalse(outcome.save_allowed)
            self.assertIsNotNone(outcome.warning)
            with self.assertRaises(PermissionError):
                store.save(outcome.config)
            restored_text = path.read_text(encoding="utf-8")
        self.assertEqual(restored_text, original_text)

    def test_ambiguous_schema_identifiers_disable_saving(self):
        for schema in (True, 1.5, "1.5", "malformed"):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                original_text = json.dumps({
                    "schema_version": schema,
                    "keep": "unchanged",
                })
                path.write_text(original_text, encoding="utf-8")
                store = ConfigStore(path)
                outcome = store.load()
                self.assertFalse(outcome.save_allowed)
                self.assertEqual(outcome.warning, "Invalid configuration schema")
                with self.assertRaises(PermissionError):
                    store.save(outcome.config)
                self.assertEqual(path.read_text(encoding="utf-8"), original_text)

    def test_corrupt_json_uses_defaults_and_reports_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("not json", encoding="utf-8")
            outcome = ConfigStore(path).load()
        self.assertTrue(outcome.save_allowed)
        self.assertIsNotNone(outcome.warning)
        self.assertEqual(outcome.config, AppConfig())

    @unittest.skipUnless(
        hasattr(sys, "set_int_max_str_digits"),
        "Python integer digit limits are unavailable",
    )
    def test_oversized_json_integer_uses_defaults_and_reports_warning(self):
        previous_limit = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(640)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(
                    '{"schema_version":' + ("9" * 641) + "}",
                    encoding="utf-8",
                )
                outcome = ConfigStore(path).load()
        finally:
            sys.set_int_max_str_digits(previous_limit)
        self.assertEqual(outcome.config, AppConfig())
        self.assertTrue(outcome.save_allowed)
        self.assertEqual(outcome.warning, "Configuration load failed: ValueError")

    def test_malformed_schema_two_values_are_safely_coerced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "schema_version": 2,
                "motion": {
                    "pulse_size_px": "bad",
                    "pulse_rate_hz": 9999,
                    "ramp_mode": "Unknown",
                },
                "trigger": "NoSuchButton",
                "modifier": "NoSuchButton",
                "hotkey_vk": 9999,
                "selected_preset": "Strong Shake",
            }), encoding="utf-8")
            config = ConfigStore(path).load().config
        self.assertEqual(config.motion, MotionSettings(2.0, 60.0, "Smooth"))
        self.assertEqual(config.trigger, "Left")
        self.assertEqual(config.modifier, "None")
        self.assertEqual(config.hotkey_vk, 255)
        self.assertEqual(config.selected_preset, "Custom")

    def test_schema_two_list_preset_uses_custom_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "schema_version": 2,
                "selected_preset": ["Strong"],
            }), encoding="utf-8")
            try:
                config = ConfigStore(path).load().config
            except TypeError:
                config = None
        self.assertIsNotNone(config)
        self.assertEqual(config.selected_preset, "Custom")

    def test_schema_one_preserves_app_choices_but_migrates_motion_to_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "motion": {"motion_strength_pps": "123"},
                "trigger": "Right",
                "modifier": "Mouse4",
                "hotkey_vk": 65,
                "hotkey_name": "A",
                "selected_preset": "Strong Shake",
                "theme": "dark",
            }), encoding="utf-8")
            outcome = ConfigStore(path).load()
            document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(outcome.config.motion, MotionSettings())
        self.assertEqual(outcome.config.trigger, "Right")
        self.assertEqual(outcome.config.modifier, "Mouse4")
        self.assertEqual(outcome.config.hotkey_vk, 65)
        self.assertEqual(outcome.config.hotkey_name, "A")
        self.assertEqual(outcome.config.selected_preset, "Balanced")
        self.assertEqual(outcome.config.theme, "dark")
        self.assertEqual(outcome.config.mode, "jitter")
        self.assertEqual(outcome.config.ai, AimSettings())
        self.assertEqual(document["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
