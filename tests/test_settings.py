import json
import tempfile
import unittest
from pathlib import Path

from motion import MotionSettings
from settings import AppConfig, ConfigStore, SCHEMA_VERSION


class ConfigStoreTests(unittest.TestCase):
    def test_missing_config_returns_safe_disabled_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            outcome = ConfigStore(Path(directory) / "config.json").load()
        self.assertEqual(outcome.config.motion, MotionSettings())
        self.assertEqual(outcome.config.trigger, "Left")
        self.assertEqual(outcome.config.modifier, "None")
        self.assertEqual(outcome.config.selected_preset, "Custom")
        self.assertEqual(outcome.config.theme, "light")
        self.assertTrue(outcome.save_allowed)

    def test_schema_two_round_trip_saves_only_paired_pulse_motion(self):
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
        self.assertEqual(document["schema_version"], 2)
        self.assertEqual(document["motion"], {
            "pulse_size_px": "4", "pulse_rate_hz": "45", "ramp_mode": "Instant",
        })
        self.assertNotIn("enabled", document)
        self.assertNotIn("moving", document)
        self.assertEqual(restored, config)

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

    def test_corrupt_json_uses_defaults_and_reports_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("not json", encoding="utf-8")
            outcome = ConfigStore(path).load()
        self.assertTrue(outcome.save_allowed)
        self.assertIsNotNone(outcome.warning)
        self.assertEqual(outcome.config, AppConfig())

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
        self.assertEqual(outcome.config.selected_preset, "Custom")
        self.assertEqual(outcome.config.theme, "dark")
        self.assertEqual(document["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
