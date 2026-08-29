import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
