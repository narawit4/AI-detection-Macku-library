# Jitter Application Package Layout Design

**Date:** 2026-08-29

**Status:** Approved in chat; awaiting written-spec review

## Summary

Jitter's implementation modules will move out of the repository root into a
new `jitter_app` Python package organized by responsibility. The stable process
entry point, packaging command, launchers, resources, documentation, licenses,
and tests remain at the root. Every production and test import moves to the new
package paths; the old flat module paths intentionally stop working and no
compatibility wrapper modules remain behind.

This is a structural migration. It must not change runtime behavior, settings,
AI model contracts, target selection, motion, threading, Makcu behavior, UI
layout, persistence, dependencies, or packaged resources.

## Goals

- Make the repository root easy to scan by leaving only entry points, build
  metadata, assets, documentation, licenses, and tests there.
- Group implementation modules into clear AI, motion, device, presentation,
  and configuration boundaries.
- Preserve `python main.py`, `run_gui.bat`, `gen.bat`, and the canonical
  distribution command.
- Preserve lazy imports in `main.py`, especially the isolated
  `--ai-runtime-self-check` path.
- Centralize bundled model and sound resource discovery so moving modules deeper
  cannot redirect resource lookups.
- Keep all existing public classes, functions, immutable data contracts, and
  runtime behavior unchanged apart from their import paths.
- Keep recursive source discovery, compilation review, DirectML DLL inclusion,
  release materials, and bundled-only model packaging correct.
- Update the full hardware-free test suite and documentation to use and describe
  the new layout.

## Non-Goals

- Do not split or redesign the internals of `ui.py`, `makcu_service.py`, or any
  other implementation module during this migration.
- Do not reorganize the `tests/` directory into subpackages.
- Do not move `main.py`, `distribution_metadata.py`, `run_gui.bat`, `gen.bat`,
  `nuitka-package.config.yml`, or `requirements.txt` away from the root.
- Do not move or rename `models/`, `sound/`, `licenses/`, `docs/`, or tests.
- Do not preserve old imports such as `import ai_detection` or
  `from settings import ConfigStore` through wrapper modules or `sys.modules`
  aliases.
- Do not change UI, motion, AI targeting, Adaptive Zoom, overlay, cadence,
  configuration schema, logging, shutdown, or model-selection behavior.
- Do not add dependencies, `pyproject.toml`, an installer, training, profiles,
  downloads, or another model runtime.
- Do not copy, move, persist, package, or commit any external `.onnx` model.
- Do not run Nuitka unless the user separately requests a confirmed build.

## Target Repository Layout

```text
Jitter/
|-- main.py
|-- distribution_metadata.py
|-- run_gui.bat
|-- gen.bat
|-- nuitka-package.config.yml
|-- requirements.txt
|-- README.md
|-- AGENTS.md
|-- LICENSE
|-- THIRD_PARTY_NOTICES.md
|-- jitter_app/
|   |-- __init__.py
|   |-- resources.py
|   |-- ai/
|   |   |-- __init__.py
|   |   |-- capture.py
|   |   |-- detection.py
|   |   |-- model_selection.py
|   |   |-- service.py
|   |   |-- targeting.py
|   |   |-- tracking.py
|   |   |-- resize.py
|   |   |-- yolo.py
|   |   `-- zoom.py
|   |-- motion/
|   |   |-- __init__.py
|   |   |-- engine.py
|   |   `-- combined.py
|   |-- device/
|   |   |-- __init__.py
|   |   |-- makcu.py
|   |   |-- hotkeys.py
|   |   `-- display_timing.py
|   |-- presentation/
|   |   |-- __init__.py
|   |   |-- ui.py
|   |   |-- widgets.py
|   |   |-- overlay.py
|   |   `-- sound.py
|   `-- config/
|       |-- __init__.py
|       `-- store.py
|-- models/
|-- sound/
|-- licenses/
|-- docs/
`-- tests/
```

Package `__init__.py` files contain package documentation only. They do not
re-export implementation symbols or eagerly import optional/runtime modules.
That keeps ownership explicit and prevents importing `jitter_app` from loading
Tkinter, ONNX Runtime, DXCam, pygame, or Makcu.

## Exact Module Migration

| Existing module | New module |
|---|---|
| `ai_capture.py` | `jitter_app/ai/capture.py` |
| `ai_detection.py` | `jitter_app/ai/detection.py` |
| `ai_model_selection.py` | `jitter_app/ai/model_selection.py` |
| `ai_service.py` | `jitter_app/ai/service.py` |
| `ai_targeting.py` | `jitter_app/ai/targeting.py` |
| `ai_tracking.py` | `jitter_app/ai/tracking.py` |
| `image_resize.py` | `jitter_app/ai/resize.py` |
| `ai_yolo.py` | `jitter_app/ai/yolo.py` |
| `ai_zoom.py` | `jitter_app/ai/zoom.py` |
| `motion.py` | `jitter_app/motion/engine.py` |
| `combined_motion.py` | `jitter_app/motion/combined.py` |
| `makcu_service.py` | `jitter_app/device/makcu.py` |
| `hotkeys.py` | `jitter_app/device/hotkeys.py` |
| `display_timing.py` | `jitter_app/device/display_timing.py` |
| `ui.py` | `jitter_app/presentation/ui.py` |
| `liquid_widgets.py` | `jitter_app/presentation/widgets.py` |
| `overlay.py` | `jitter_app/presentation/overlay.py` |
| `sound_service.py` | `jitter_app/presentation/sound.py` |
| `settings.py` | `jitter_app/config/store.py` |

The migration uses `git mv` so file history remains traceable. After the move,
the only root Python application/build entry modules are `main.py` and
`distribution_metadata.py`. All old implementation paths are absent.

## Dependency Boundaries and Import Rules

Modules in the same package use explicit relative imports. Cross-package
dependencies use absolute `jitter_app` imports. Tests always import the public
module that owns a symbol rather than relying on package-level re-exports.

The dependency directions remain behaviorally identical:

- `jitter_app.ai` contains pure targeting/tracking/zoom/resize/YOLO logic and
  the ONNX/DXCam service boundaries.
- `jitter_app.motion` contains pure Jitter settings/engine and source
  composition. It may consume immutable AI target/settings types where the
  existing combined engine already does so.
- `jitter_app.device` contains Windows and Makcu boundaries and may consume
  motion/AI engine types.
- `jitter_app.presentation` owns Tk UI, widgets, overlay, and sound playback;
  it coordinates the other packages without being imported by them.
- `jitter_app.config` owns validated user configuration and may consume the
  immutable AI and motion settings types it serializes.
- `jitter_app.resources` depends only on `pathlib` and is safe to import from
  detection, presentation, configuration, entrypoint, and tests.

Representative imports are:

```python
from .targeting import Detection
from .yolo import RAW_CANDIDATE_COUNTS, decode_single_class_yolo
from jitter_app.motion.engine import MotionSettings
from jitter_app.device.makcu import MakcuService
```

`main.py` remains deliberately lazy:

```python
from jitter_app.config.store import ConfigStore
from jitter_app.presentation.ui import JitterApp
from jitter_app.ai.detection import OnnxDetector, model_resource_path
```

Those imports remain inside the existing lazy factory/self-check functions so
the self-check does not import the normal Tk application or settings path.

No module alias, fallback import, or compatibility shim accepts an old root
module path. Tests explicitly prove representative old modules are unavailable.

## Resource and User-Data Paths

`jitter_app/resources.py` is the sole definition of the source or one-file
bundle root:

```python
def bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bundled_model_path() -> Path:
    return bundle_root() / "models" / "all_games_320.onnx"


def sound_directory() -> Path:
    return bundle_root() / "sound"
```

The name `bundle_root` describes both the repository root during a source run
and Nuitka's extraction root during a one-file run. It is independent of the
current working directory.

`jitter_app.ai.detection.model_resource_path()` delegates to
`bundled_model_path()` so its existing public callable and return contract stay
available at the new module path. `jitter_app.presentation.ui` uses
`sound_directory()` for the default `ToggleSoundPlayer`.

Moving configuration into `jitter_app/config/store.py` must not move user data.
In source mode, `runtime_base_dir()` returns `bundle_root()` rather than the
configuration package directory. In compiled mode it retains the existing
`__compiled__.containing_dir`, executable-directory fallback, validation, and
failure behavior. Therefore `config.json`, `config.json.bak`, and `app.log`
remain beside `main.py` in source runs or beside the executable in releases.

Resource helpers do not create directories, copy assets, search alternate
locations, or accept external-model paths. Missing packaged resources continue
to fail through the existing authored initialization/self-check paths.

## Entrypoints and Runtime Behavior

`python main.py`, double-clicking `run_gui.bat`, and the no-argument confirmed
`gen.bat` flow stay unchanged. `main.py` remains the sole process entry point
and continues to own mutex creation, logging setup, the exact
`--ai-runtime-self-check` dispatch, and shutdown.

No command-line flag or import side effect is added. `distribution_metadata.py`
remains directly runnable at the root for `--help`, `--review-json`, and the
explicit build flow.

The move must not alter:

- thread ownership or generation barriers;
- Makcu callbacks, reconnect, trigger/modifier gates, or movement;
- nearest-current-frame target selection or detector-order tie breaks;
- legacy/raw ONNX contracts, response curve, servo, or Adaptive Zoom;
- overlay visibility, capture exclusion, colors, or head-box behavior;
- configuration schema, atomic writes, unsupported-schema protection, or
  runtime-only fields;
- error text except where a module path appears only in detailed diagnostics.

## Packaging and Distribution Metadata

`distribution_metadata.discover_application_sources()` already discovers live
Python sources recursively. After migration, its reviewed `compile_targets`
must contain:

- `main.py` and `distribution_metadata.py`;
- every `jitter_app/**/*.py` module, including each `__init__.py`;
- no removed flat implementation path;
- no test, worktree, build output, or generated source.

The local-import inventory treats `jitter_app` as one local import root, so it
must not appear as an external dependency. The active third-party runtime roots
and exact pinned dependency/license inventory remain unchanged.

`nuitka-package.config.yml` changes its module key from:

```yaml
- module-name: 'ai_detection'
```

to:

```yaml
- module-name: 'jitter_app.ai.detection'
```

The configuration's reviewed SHA-256 constant and corresponding tests are
updated to the new file bytes. DirectML DLL source, destination, and hash remain
unchanged.

The Nuitka entry module stays `main.py`, and the canonical model data option
stays exactly:

```text
--include-data-files=models/all_games_320.onnx=models/all_games_320.onnx
```

Release materials remain `LICENSE`, `THIRD_PARTY_NOTICES.md`, and the complete
`licenses/` directory. External models remain untracked runtime inputs and are
never discovered as application source or package data.

## Tests and Mock Paths

Tests remain directly under `tests/` and retain their existing behavioral
grouping. Their production imports and patch targets change to the exact new
owners, for example:

```python
from jitter_app.ai.detection import OnnxDetector
from jitter_app.config.store import ConfigStore
mock.patch("jitter_app.ai.service.model_resource_path", ...)
```

Tests that execute a copied `main.py` in an isolated directory create the
minimal package fixture required by the lazy path:

```text
jitter_app/
|-- __init__.py
`-- ai/
    |-- __init__.py
    `-- detection.py
```

They do not restore an `ai_detection.py` compatibility file. Separate structure
tests assert:

- every exact old implementation file is absent from the root;
- every exact new package module exists;
- root application/build Python files are exactly `main.py` and
  `distribution_metadata.py`;
- importing representative old paths raises `ModuleNotFoundError` after caches
  are invalidated and the old names are removed from `sys.modules`;
- importing `jitter_app` alone does not eagerly load Tkinter, ONNX Runtime,
  DXCam, pygame, or Makcu modules;
- configuration, model, and sound resource paths still resolve to root assets;
- compile and package inventories match the new paths and bundled-only model
  policy.

All pre-existing behavioral tests continue to run. Test assertions change only
where the subject is a module path, source inventory, resource location, or
packaging module name.

## Migration Sequence

Implementation follows test-driven development:

1. Add failing structure/resource/package tests for the approved target layout.
2. Add the package directories, documentation-only `__init__.py` files, and
   pure resource helper.
3. Move implementation modules with `git mv` and update production imports.
4. Update all test imports, patch targets, and isolated entrypoint fixtures.
5. Update source/package inventories, the Nuitka module key/hash, README, and
   AGENTS guidance.
6. Remove generated `__pycache__` directories only as untracked cache cleanup;
   do not alter build output or user data.
7. Verify the complete repository at the new paths before integration.

The structural move may be one atomic implementation commit so the branch does
not intentionally retain a commit with half the modules at old paths. Review can
separate the package/resource behavior from documentation through later commits
without preserving a broken intermediate tree.

## Verification

Run from the repository root:

```powershell
python -m py_compile main.py distribution_metadata.py (Get-ChildItem jitter_app -Recurse -Filter *.py | ForEach-Object FullName)
python -m unittest discover -s tests -v
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
python .\main.py --ai-runtime-self-check
python .\distribution_metadata.py --review-json
git diff --check
```

The exact compile command used in automated review must be constructed as an
argument array rather than passing PowerShell expression text to Python. The
distribution review JSON is the authoritative compile-target list.

When the user-owned Apex model exists, run one zero-frame inference through
`jitter_app.ai.detection.OnnxDetector` and verify input size 640,
`DmlExecutionProvider`, class `0`-only output, and identical file size/SHA-256
before and after. Keep the model untracked and unstaged.

Do not run Nuitka for this structural change. Connected Makcu verification
remains required before claiming hardware behavior, and a separately authorized
packaged build remains required before claiming binary-release success.

## Compatibility and Rollout

The user-facing launch commands, settings files, bundled model, sound assets,
and release layout remain compatible. Python import paths are intentionally
breaking for internal/test consumers: only `jitter_app.*` paths are supported
after migration.

The application starts with the same bundled model and safe defaults. A failure
to resolve a moved import or root resource is caught by compile, unit,
self-check, and distribution-review gates before handoff. No migration of user
configuration or external model data is needed.
