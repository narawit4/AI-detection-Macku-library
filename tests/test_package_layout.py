import json
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


OLD_FILES = {
    "ai_capture.py", "ai_detection.py", "ai_model_selection.py",
    "ai_service.py", "ai_targeting.py", "ai_tracking.py", "image_resize.py",
    "ai_yolo.py", "ai_zoom.py", "motion.py", "combined_motion.py",
    "makcu_service.py", "hotkeys.py", "display_timing.py", "ui.py",
    "liquid_widgets.py", "overlay.py", "sound_service.py", "settings.py",
}

NEW_FILES = {
    "jitter_app/__init__.py", "jitter_app/resources.py",
    "jitter_app/ai/__init__.py", "jitter_app/ai/capture.py",
    "jitter_app/ai/detection.py", "jitter_app/ai/model_selection.py",
    "jitter_app/ai/service.py", "jitter_app/ai/targeting.py",
    "jitter_app/ai/tracking.py", "jitter_app/ai/resize.py",
    "jitter_app/ai/yolo.py", "jitter_app/ai/zoom.py",
    "jitter_app/motion/__init__.py", "jitter_app/motion/engine.py",
    "jitter_app/motion/combined.py", "jitter_app/device/__init__.py",
    "jitter_app/device/makcu.py", "jitter_app/device/hotkeys.py",
    "jitter_app/device/display_timing.py",
    "jitter_app/presentation/__init__.py",
    "jitter_app/presentation/ui.py", "jitter_app/presentation/widgets.py",
    "jitter_app/presentation/overlay.py", "jitter_app/presentation/sound.py",
    "jitter_app/config/__init__.py", "jitter_app/config/store.py",
}


class PackageFoundationTests(unittest.TestCase):
    def test_resource_helpers_resolve_root_assets(self):
        from jitter_app.resources import (
            bundle_root,
            bundled_model_path,
            sound_directory,
        )

        self.assertEqual(bundle_root(), ROOT)
        self.assertEqual(
            bundled_model_path(), ROOT / "models" / "all_games_320.onnx"
        )
        self.assertEqual(sound_directory(), ROOT / "sound")

    def test_importing_package_has_no_runtime_stack_side_effects(self):
        code = """
import json
import sys
import jitter_app
blocked = ('tkinter', 'onnxruntime', 'dxcam', 'pygame', 'makcu')
print(json.dumps([name for name in blocked if name in sys.modules]))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        self.assertEqual(json.loads(completed.stdout), [])


class PackageStructureTests(unittest.TestCase):
    def test_exact_new_modules_exist_and_old_files_are_absent(self):
        self.assertEqual(
            {path for path in NEW_FILES if not (ROOT / path).is_file()}, set()
        )
        self.assertEqual(
            {path for path in OLD_FILES if (ROOT / path).exists()}, set()
        )

    def test_root_python_entrypoints_are_exact(self):
        self.assertEqual(
            {path.name for path in ROOT.glob("*.py")},
            {"main.py", "distribution_metadata.py"},
        )

    def test_representative_flat_imports_are_unavailable(self):
        code = """
import importlib.util
import json
names = ('ai_detection', 'motion', 'makcu_service', 'ui', 'settings')
print(json.dumps([name for name in names if importlib.util.find_spec(name)]))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        )
        self.assertEqual(json.loads(completed.stdout), [])
