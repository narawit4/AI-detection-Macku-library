import json
import tempfile
import unittest
from dataclasses import replace
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

    def test_valid_config_round_trips_without_runtime_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            store = ConfigStore(path)
            config = AppConfig(
                motion=replace(MotionSettings(), strength_pps=123.0),
                trigger="Mouse4", modifier="Right", hotkey_vk=0x77,
                hotkey_name="F8", selected_preset="Custom", theme="dark",
            )
            store.save(config)
            document = json.loads(path.read_text(encoding="utf-8"))
            outcome = store.load()
        self.assertEqual(document["schema_version"], SCHEMA_VERSION)
        self.assertEqual(document["theme"], "dark")
        self.assertNotIn("enabled", document)
        self.assertNotIn("moving", document)
        self.assertEqual(outcome.config, config)

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
            second = AppConfig(selected_preset="Extreme")
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

    def test_invalid_values_are_safely_coerced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "motion": {"motion_strength_pps": "bad"},
                "trigger": "NoSuchButton",
                "modifier": "NoSuchButton",
                "hotkey_vk": 9999,
            }), encoding="utf-8")
            config = ConfigStore(path).load().config
        self.assertEqual(config.motion, MotionSettings())
        self.assertEqual(config.trigger, "Left")
        self.assertEqual(config.modifier, "None")
        self.assertEqual(config.hotkey_vk, 255)


if __name__ == "__main__":
    unittest.main()
