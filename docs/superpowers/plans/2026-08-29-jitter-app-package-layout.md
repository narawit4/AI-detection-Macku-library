# Jitter Application Package Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every Jitter implementation module into the responsibility-based `jitter_app` package while preserving runtime behavior, root entrypoints, resource locations, and packaging guarantees.

**Architecture:** Introduce documentation-only packages plus one pure resource-path module, then perform the implementation-module relocation as one atomic green commit so no supported commit contains half-migrated imports. Keep `main.py` and `distribution_metadata.py` at the repository root, use relative imports within a subpackage and absolute `jitter_app.*` imports across package boundaries, and intentionally remove every old flat import path.

**Tech Stack:** Python 3 on Windows, Tkinter/ttk, unittest, ONNX Runtime DirectML, DXCam, NumPy, Makcu, pygame-ce, Nuitka package configuration, Git.

**Spec:** `docs/superpowers/specs/2026-08-29-jitter-app-package-layout-design.md`

## Global Constraints

- This is a structural migration only: do not change UI, motion, AI targeting, Adaptive Zoom, overlay, cadence, configuration schema, logging, shutdown, model contracts, or model-selection behavior.
- Keep root entrypoints `main.py` and `distribution_metadata.py`, launchers `run_gui.bat` and `gen.bat`, and the exact bundled model option `--include-data-files=models/all_games_320.onnx=models/all_games_320.onnx`.
- Do not add dependencies, compatibility wrappers, package re-exports, fallback imports, aliases, `pyproject.toml`, training, profiles, downloads, copying, persistence, or another model runtime.
- Keep `models/`, `sound/`, `licenses/`, `docs/`, and `tests/` at the repository root.
- Never stage, edit, move, copy, or delete untracked external `.onnx` files. In particular preserve `models/Apex_20k_pictures_640.onnx` byte-for-byte and untracked.
- Preserve source-mode `config.json`, `config.json.bak`, and `app.log` beside `main.py`; preserve the existing compiled-directory behavior.
- Keep Tk access on the main thread and retain all existing worker cancellation, generation, locking, and lifecycle semantics.
- Runtime dependencies remain exactly pinned; do not add Torch, Ultralytics, OpenCV, Pillow, or alternate runtimes.
- Do not edit generated `build-output/`, `dist/`, `*.build/`, `*.dist/`, `__pycache__/`, or `app.log` as source.
- Do not run Nuitka. A packaged build requires a separate explicit user request.
- Follow TDD: observe the focused test fail, implement minimally, run the focused test, then run the full hardware-free suite before each implementation commit.

## Target File Map

Create these package-only files:

```text
jitter_app/__init__.py                    package documentation only
jitter_app/resources.py                   bundle_root/model/sound path helpers
jitter_app/ai/__init__.py                 package documentation only
jitter_app/motion/__init__.py             package documentation only
jitter_app/device/__init__.py             package documentation only
jitter_app/presentation/__init__.py       package documentation only
jitter_app/config/__init__.py             package documentation only
```

Move implementation files without splitting or redesigning them:

```text
ai_capture.py          -> jitter_app/ai/capture.py
ai_detection.py        -> jitter_app/ai/detection.py
ai_model_selection.py  -> jitter_app/ai/model_selection.py
ai_service.py          -> jitter_app/ai/service.py
ai_targeting.py        -> jitter_app/ai/targeting.py
ai_tracking.py         -> jitter_app/ai/tracking.py
image_resize.py        -> jitter_app/ai/resize.py
ai_yolo.py             -> jitter_app/ai/yolo.py
ai_zoom.py             -> jitter_app/ai/zoom.py
motion.py               -> jitter_app/motion/engine.py
combined_motion.py      -> jitter_app/motion/combined.py
makcu_service.py        -> jitter_app/device/makcu.py
hotkeys.py              -> jitter_app/device/hotkeys.py
display_timing.py       -> jitter_app/device/display_timing.py
ui.py                   -> jitter_app/presentation/ui.py
liquid_widgets.py       -> jitter_app/presentation/widgets.py
overlay.py              -> jitter_app/presentation/overlay.py
sound_service.py        -> jitter_app/presentation/sound.py
settings.py             -> jitter_app/config/store.py
```

Keep these responsibilities at the root:

```text
main.py                  process entrypoint, mutex, logging, lazy startup/self-check
distribution_metadata.py distribution review and explicit build orchestration
```

---

### Task 1: Add the lazy package and centralized resource paths

**Files:**
- Create: `jitter_app/__init__.py`
- Create: `jitter_app/resources.py`
- Create: `jitter_app/ai/__init__.py`
- Create: `jitter_app/motion/__init__.py`
- Create: `jitter_app/device/__init__.py`
- Create: `jitter_app/presentation/__init__.py`
- Create: `jitter_app/config/__init__.py`
- Create: `tests/test_package_layout.py`
- Modify: `ai_detection.py`
- Modify: `ui.py`
- Modify: `settings.py`
- Modify: `tests/test_entrypoints.py`
- Modify: `tests/test_distribution_metadata.py`

**Interfaces:**
- Consumes: repository-root `models/all_games_320.onnx`, `sound/`, and the existing `model_resource_path() -> Path` and `runtime_base_dir() -> Path` contracts.
- Produces: `bundle_root() -> Path`, `bundled_model_path() -> Path`, and `sound_directory() -> Path` in `jitter_app.resources`; documentation-only packages with no eager imports.

- [ ] **Step 1: Write the failing package/resource tests**

Create `tests/test_package_layout.py` with the exact foundation checks below. At this task, do not assert that implementation modules have moved yet.

```python
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
```

- [ ] **Step 2: Run the new test and verify the expected RED state**

Run:

```powershell
python -m unittest tests.test_package_layout -v
```

Expected: FAIL or ERROR with `ModuleNotFoundError: No module named 'jitter_app'`.

- [ ] **Step 3: Add documentation-only package files and the pure helper**

Each `__init__.py` contains only the corresponding package docstring below and no imports:

```python
# jitter_app/__init__.py
"""Jitter application implementation package."""

# jitter_app/ai/__init__.py
"""AI capture, detection, targeting, and inference services."""

# jitter_app/motion/__init__.py
"""Pure Jitter and combined movement engines."""

# jitter_app/device/__init__.py
"""Windows input, display, and Makcu device boundaries."""

# jitter_app/presentation/__init__.py
"""Tk presentation, overlay, widgets, and sound services."""

# jitter_app/config/__init__.py
"""Validated persistent configuration."""
```

Implement `jitter_app/resources.py` exactly as follows:

```python
"""Resolve Jitter resources without depending on the working directory."""

from pathlib import Path


def bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bundled_model_path() -> Path:
    return bundle_root() / "models" / "all_games_320.onnx"


def sound_directory() -> Path:
    return bundle_root() / "sound"
```

- [ ] **Step 4: Route the existing flat modules through the new helper without changing their public behavior**

Make these exact substitutions:

```python
# ai_detection.py
from jitter_app.resources import bundled_model_path

def model_resource_path() -> Path:
    return bundled_model_path()

# ui.py
from jitter_app.resources import sound_directory

ToggleSoundPlayer(
    sound_directory(),
    enabled=self.config.sound_enabled,
    volume=self.config.sound_volume,
)

# settings.py
from jitter_app.resources import bundle_root

# Keep every compiled-mode branch unchanged. Replace only the source fallback:
source_dir = bundle_root()
```

Remove `pathlib.Path` imports only where they become unused; do not alter compiled fallback code that still needs `Path`.

- [ ] **Step 5: Update the two exact compile-target assertions for the temporary green foundation state**

In both `tests/test_entrypoints.py` and `tests/test_distribution_metadata.py`, retain the existing flat implementation paths and add these seven paths to the exact expected set:

```python
{
    "jitter_app/__init__.py",
    "jitter_app/resources.py",
    "jitter_app/ai/__init__.py",
    "jitter_app/motion/__init__.py",
    "jitter_app/device/__init__.py",
    "jitter_app/presentation/__init__.py",
    "jitter_app/config/__init__.py",
}
```

Do not weaken the assertion by deriving expected values from source discovery.

- [ ] **Step 6: Run focused resource, configuration, entrypoint, and distribution tests**

Run:

```powershell
python -m unittest tests.test_package_layout tests.test_ai_detection tests.test_settings tests.test_entrypoints tests.test_distribution_metadata -v
```

Expected: PASS, including source resource paths at the repository root and unchanged compiled-mode path tests.

- [ ] **Step 7: Run the full hardware-free suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: PASS with no behavior-test regressions.

- [ ] **Step 8: Commit the package foundation**

```powershell
git add jitter_app tests/test_package_layout.py tests/test_entrypoints.py tests/test_distribution_metadata.py ai_detection.py ui.py settings.py
git commit -m "refactor: add application package foundation"
```

---

### Task 2: Atomically move all implementation modules and imports

**Files:**
- Move: all 19 implementation files in the Target File Map
- Modify: `main.py`
- Modify: every moved module containing a local import
- Modify: all `tests/test_*.py` files that import or patch a moved module
- Modify: `tests/test_package_layout.py`
- Modify: `tests/test_entrypoints.py`
- Modify: `tests/test_distribution_metadata.py`
- Modify: `distribution_metadata.py`
- Modify: `nuitka-package.config.yml`

**Interfaces:**
- Consumes: Task 1's `jitter_app.resources` functions and all existing public classes/functions at their current flat-module owners.
- Produces: the same public classes/functions at the exact `jitter_app.*` owners in the Target File Map; lazy imports in `main.py`; no root implementation modules or compatibility paths; Nuitka configuration for `jitter_app.ai.detection` with SHA-256 `E2D715C37C2EF10D3195F1DC05997F322E9E2F136755D9860664F10E4A48D2DE`.

- [ ] **Step 1: Extend the structure test to describe the complete target layout**

Append these constants and tests to `tests/test_package_layout.py`:

```python
import importlib.util


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
```

- [ ] **Step 2: Run the structure tests and verify the expected RED state**

Run:

```powershell
python -m unittest tests.test_package_layout.PackageStructureTests -v
```

Expected: FAIL because the new implementation files do not exist and the old files still do.

- [ ] **Step 3: Move all implementation files with Git history**

Run exactly these moves from the repository root:

```powershell
git mv ai_capture.py jitter_app/ai/capture.py
git mv ai_detection.py jitter_app/ai/detection.py
git mv ai_model_selection.py jitter_app/ai/model_selection.py
git mv ai_service.py jitter_app/ai/service.py
git mv ai_targeting.py jitter_app/ai/targeting.py
git mv ai_tracking.py jitter_app/ai/tracking.py
git mv image_resize.py jitter_app/ai/resize.py
git mv ai_yolo.py jitter_app/ai/yolo.py
git mv ai_zoom.py jitter_app/ai/zoom.py
git mv motion.py jitter_app/motion/engine.py
git mv combined_motion.py jitter_app/motion/combined.py
git mv makcu_service.py jitter_app/device/makcu.py
git mv hotkeys.py jitter_app/device/hotkeys.py
git mv display_timing.py jitter_app/device/display_timing.py
git mv ui.py jitter_app/presentation/ui.py
git mv liquid_widgets.py jitter_app/presentation/widgets.py
git mv overlay.py jitter_app/presentation/overlay.py
git mv sound_service.py jitter_app/presentation/sound.py
git mv settings.py jitter_app/config/store.py
```

Before and after this step, run `git status --short models` and confirm every external model remains `??` at its existing path.

- [ ] **Step 4: Convert production imports to their exact new owners**

Use relative imports inside one subpackage and absolute imports across package boundaries. Apply this complete ownership table to moved modules and `main.py`:

```text
ai_capture          -> jitter_app.ai.capture
ai_detection        -> jitter_app.ai.detection
ai_model_selection  -> jitter_app.ai.model_selection
ai_service          -> jitter_app.ai.service
ai_targeting        -> jitter_app.ai.targeting
ai_tracking         -> jitter_app.ai.tracking
image_resize        -> jitter_app.ai.resize
ai_yolo             -> jitter_app.ai.yolo
ai_zoom             -> jitter_app.ai.zoom
motion              -> jitter_app.motion.engine
combined_motion     -> jitter_app.motion.combined
makcu_service       -> jitter_app.device.makcu
hotkeys             -> jitter_app.device.hotkeys
display_timing      -> jitter_app.device.display_timing
ui                  -> jitter_app.presentation.ui
liquid_widgets      -> jitter_app.presentation.widgets
overlay             -> jitter_app.presentation.overlay
sound_service       -> jitter_app.presentation.sound
settings            -> jitter_app.config.store
```

The representative production imports must end in this form:

```python
# jitter_app/ai/detection.py
from .targeting import Detection
from .yolo import RAW_CANDIDATE_COUNTS, decode_single_class_yolo
from .resize import resize_rgb_bilinear
from jitter_app.resources import bundled_model_path

# jitter_app/ai/service.py
from .capture import DxcamCapture
from .detection import OnnxDetector, model_resource_path
from .targeting import AimSettings, Detection, DetectionFrameSnapshot

# jitter_app/motion/combined.py
from jitter_app.ai.targeting import AimMovementEngine, AimSettings, TargetSnapshot
from .engine import MotionSettings, PairedPulseEngine

# jitter_app/device/makcu.py
from jitter_app.ai.targeting import AimMovementEngine, AimSettings, TargetSnapshot
from jitter_app.motion.combined import CombinedMotionEngine, MotionSources
from jitter_app.motion.engine import MotionSettings, PairedPulseEngine

# jitter_app/presentation/ui.py
from jitter_app.ai.service import AiEvent, AiService
from jitter_app.device.makcu import MakcuService, ServiceEvent
from jitter_app.motion.engine import MotionSettings
from jitter_app.config.store import AppConfig, ConfigStore, normalize_overlay_color
from .widgets import LiquidIconButton, LiquidNavigation, LiquidSlider
from .sound import ToggleSoundPlayer
from .overlay import DetectionOverlay, OverlaySetupError
```

Keep `main.py` imports inside their existing functions, changing only their owners:

```python
from jitter_app.config.store import runtime_base_dir as resolve_runtime_base_dir
from jitter_app.config.store import ConfigStore as ConfigStoreType
from jitter_app.presentation.ui import JitterApp as JitterAppType
from jitter_app.ai.detection import OnnxDetector, model_resource_path
```

Do not add imports to any `__init__.py`.

- [ ] **Step 5: Update every test import, patch target, logger name, and isolated fixture**

Apply the ownership table from Step 4 to all `tests/test_*.py` imports. Update string targets exactly:

```text
ai_service.*          -> jitter_app.ai.service.*
ai_model_selection.*  -> jitter_app.ai.model_selection.*
ui.*                  -> jitter_app.presentation.ui.*
sound_service.*       -> jitter_app.presentation.sound.*
makcu_service.*       -> jitter_app.device.makcu.*
```

This includes `mock.patch(...)`, `assertLogs(...)`, `importlib.import_module(...)`, `importlib.util.find_spec(...)`, and module aliases such as:

```python
import jitter_app.ai.resize as image_resize
import jitter_app.presentation.widgets as liquid_widgets
import jitter_app.config.store as settings_module
```

Add this helper near `ROOT` in `tests/test_entrypoints.py`:

```python
def write_isolated_detection(isolated: Path, contents: str) -> None:
    package = isolated / "jitter_app"
    ai_package = package / "ai"
    ai_package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        '"""Test package."""\n', encoding="utf-8"
    )
    (ai_package / "__init__.py").write_text(
        '"""Test AI package."""\n', encoding="utf-8"
    )
    (ai_package / "detection.py").write_text(contents, encoding="utf-8")
```

Replace the failure fixture write with:

```python
write_isolated_detection(isolated, f"raise ImportError({secret!r})\n")
```

Replace the successful fixture write with:

```python
write_isolated_detection(
    isolated,
    "from pathlib import Path\n"
    "class OnnxDetector:\n"
    "    def __init__(self, _path):\n"
    "        self.provider = 'DmlExecutionProvider'\n"
    "def model_resource_path():\n"
    f"    return Path({str(model)!r})\n",
)
```

Do not create an old root `ai_detection.py`.

Update both source-fallback assertions in `tests/test_settings.py` because the owning module is now nested. They must compare `runtime_base_dir()` with the repository/bundle root, not with `Path(settings_module.__file__).parent`:

```python
from jitter_app.resources import bundle_root

self.assertEqual(runtime_base_dir(), bundle_root())
```

Keep the compiled containing-directory and executable fallback assertions unchanged.

- [ ] **Step 6: Update distribution and Nuitka ownership without changing package contents**

Make these exact changes:

```python
# distribution_metadata.py
_NUITKA_PACKAGE_CONFIG_SHA256 = (
    "E2D715C37C2EF10D3195F1DC05997F322E9E2F136755D9860664F10E4A48D2DE"
)

# returned NuitkaPackageConfiguration
module="jitter_app.ai.detection"

# sound pairing check
sound_source = root / "jitter_app" / "presentation" / "sound.py"
```

Change the YAML key only:

```yaml
- module-name: 'jitter_app.ai.detection'
```

Update `tests/test_distribution_metadata.py` to parse `jitter_app.ai.detection`, expect the new module string and exact SHA above, and detect sound at `jitter_app/presentation/sound.py`. Keep the DirectML source, destination, and DLL hash identical.

- [ ] **Step 7: Replace temporary compile inventories with the exact final inventory**

In `tests/test_entrypoints.py` and `tests/test_distribution_metadata.py`, assert this exact set:

```python
{
    "main.py", "distribution_metadata.py",
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
```

The sound asset option and `pygame` runtime inventory remain conditional on `jitter_app/presentation/sound.py` plus `sound/`; the bundled model data option remains literal and unchanged.

- [ ] **Step 8: Scan for stale flat imports and paths before running tests**

Run:

```powershell
rg -n '^(from|import) (ai_|motion|combined_motion|makcu_service|hotkeys|display_timing|ui|liquid_widgets|overlay|sound_service|settings|image_resize)' --glob '*.py'
rg -n '(ai_service|ai_model_selection|makcu_service|sound_service|ui)\.' tests --glob '*.py'
```

Expected: no matches. References inside `OLD_FILES`, migration documentation, and negative import tests are allowed; production/test import statements and patch strings are not.

- [ ] **Step 9: Run focused structure, entrypoint, packaging, and path tests**

Run:

```powershell
python -m unittest tests.test_package_layout tests.test_entrypoints tests.test_distribution_metadata tests.test_settings tests.test_ai_detection tests.test_sound_service -v
```

Expected: PASS. Verify the isolated self-check tests do not create config/log files and the package import test does not load runtime stacks.

- [ ] **Step 10: Compile the exact new source tree through an argument array**

Run:

```powershell
$jitterSources = @('main.py', 'distribution_metadata.py') + @(Get-ChildItem -LiteralPath 'jitter_app' -Recurse -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
python -m py_compile @jitterSources
```

Expected: exit code 0. Do not pass the PowerShell expression itself as one Python argument.

- [ ] **Step 11: Run the full hardware-free suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: PASS with the same behavior coverage as before the move.

- [ ] **Step 12: Commit the atomic module migration**

Inspect `git status --short`, confirm no external model is staged, then run:

```powershell
git add main.py distribution_metadata.py nuitka-package.config.yml jitter_app tests
git commit -m "refactor: organize implementation modules by responsibility"
```

---

### Task 3: Update repository guidance for the supported package paths

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `tests/test_entrypoints.py`

**Interfaces:**
- Consumes: Task 2's exact package layout and supported launch/build commands.
- Produces: Thai user documentation and repository instructions that mention only supported production paths and compile the recursive package source set.

- [ ] **Step 1: Add a failing documentation-contract test**

Add this test to `tests/test_entrypoints.py`:

```python
def test_documentation_describes_the_supported_package_layout(self):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    for text in (readme, agents):
        self.assertIn("jitter_app/ai/detection.py", text)
        self.assertIn("jitter_app/presentation/ui.py", text)
        self.assertIn("jitter_app/config/store.py", text)
        self.assertIn("Get-ChildItem", text)

    self.assertNotIn("python -m py_compile main.py ui.py", readme)
    self.assertNotIn("python -m py_compile main.py ui.py", agents)
```

- [ ] **Step 2: Run the documentation test and verify the expected RED state**

Run:

```powershell
python -m unittest tests.test_entrypoints.EntryPointTests.test_documentation_describes_the_supported_package_layout -v
```

Expected: FAIL because README and AGENTS still describe flat modules and the old compile command.

- [ ] **Step 3: Update README and AGENTS without changing scope**

Replace the flat planned-layout lists with the exact Target File Map. Update prose references such as `image_resize.py` to `jitter_app/ai/resize.py`, and use this PowerShell verification block in both documents:

```powershell
$jitterSources = @('main.py', 'distribution_metadata.py') + @(Get-ChildItem -LiteralPath 'jitter_app' -Recurse -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
python -m py_compile @jitterSources
python -m unittest discover -s tests -v
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
```

Keep the README in Thai, preserve all user-facing instructions and screenshots, and keep the explicit statement that only `models/all_games_320.onnx` is bundled. In AGENTS, preserve every architecture, threading, scope, configuration, packaging, and hardware-verification rule; change only module paths and commands affected by the move.

- [ ] **Step 4: Run the focused documentation and distribution tests**

Run:

```powershell
python -m unittest tests.test_entrypoints tests.test_distribution_metadata -v
```

Expected: PASS.

- [ ] **Step 5: Run documentation/path scans**

Run:

```powershell
rg -n 'python -m py_compile main\.py ui\.py|`(ai_detection|ui|settings|motion|makcu_service|sound_service)\.py`' README.md AGENTS.md
```

Expected: no stale supported-path or compile-command matches. Historical paths inside the approved spec/plan are intentional and are not scanned.

- [ ] **Step 6: Run the full hardware-free suite and commit**

Run:

```powershell
python -m unittest discover -s tests -v
git diff --check
git add README.md AGENTS.md tests/test_entrypoints.py
git commit -m "docs: document organized application package"
```

Expected: tests pass, `git diff --check` exits 0, and the commit contains no model or generated file.

---

### Task 4: Perform final runtime, distribution, and user-model verification

**Files:**
- Verify only: no source file changes expected
- Protect: `models/Apex_20k_pictures_640.onnx`

**Interfaces:**
- Consumes: the completed package migration and all canonical verification entrypoints.
- Produces: recorded evidence that source compilation, unit behavior, dependencies, DirectML self-check, distribution review, and the user-owned Apex model still work without a Nuitka build or hardware claim.

- [ ] **Step 1: Capture the external Apex model identity before inference**

Run:

```powershell
$apexModel = 'C:\Users\User\Desktop\Jitter\models\Apex_20k_pictures_640.onnx'
$apexBefore = Get-Item -LiteralPath $apexModel
$apexHashBefore = (Get-FileHash -LiteralPath $apexModel -Algorithm SHA256).Hash
git status --short -- models
```

Expected: the file exists and remains listed as `?? models/Apex_20k_pictures_640.onnx`; record `$apexBefore.Length` and `$apexHashBefore` without staging it.

- [ ] **Step 2: Run the canonical source verification**

Run each command separately:

```powershell
$jitterSources = @('main.py', 'distribution_metadata.py') + @(Get-ChildItem -LiteralPath 'jitter_app' -Recurse -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
python -m py_compile @jitterSources
python -m unittest discover -s tests -v
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
git diff --check
```

Expected: every command exits 0; self-check reports `status: ok` and `provider: DmlExecutionProvider`; review JSON contains only the final compile inventory and bundled `all_games_320.onnx` model option.

- [ ] **Step 3: Run a zero-frame Apex detector initialization and inference**

Run:

```powershell
python -c "from pathlib import Path; import numpy as np; from jitter_app.ai.detection import OnnxDetector; p=Path(r'C:\Users\User\Desktop\Jitter\models\Apex_20k_pictures_640.onnx'); d=OnnxDetector(p); result=d.detect(np.zeros((320,320,3),dtype=np.uint8)); print({'input_size':d.input_size,'provider':d.provider,'classes':sorted({x.class_id for x in result}),'detections':len(result)})"
```

Expected: exit code 0, `input_size` is `640`, provider is `DmlExecutionProvider`, and `classes` is either empty or `[0]`. The 320-by-320 zero frame intentionally exercises the detector's canonical-frame-to-640 resize path and performs no write to the model.

- [ ] **Step 4: Prove the Apex file stayed byte-for-byte unchanged and untracked**

Run:

```powershell
$apexAfter = Get-Item -LiteralPath $apexModel
$apexHashAfter = (Get-FileHash -LiteralPath $apexModel -Algorithm SHA256).Hash
if ($apexAfter.Length -ne $apexBefore.Length -or $apexHashAfter -ne $apexHashBefore) { throw 'Apex model changed during verification' }
git status --short -- models
git diff --cached --name-only
```

Expected: size and SHA-256 are identical, the Apex model remains `??`, and no `.onnx` path is staged.

- [ ] **Step 5: Review the final Git state and document verification limits**

Run:

```powershell
git status --short
git log -4 --oneline
```

Expected: only the known user-owned external models remain untracked. Report the exact test count and command results. State explicitly that Nuitka was not run and connected-Makcu behavior was not verified unless those separate checks were actually authorized and performed.

No commit is created in this task because it is verification-only.
