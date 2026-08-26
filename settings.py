"""Independent, schema-aware configuration for the Jitter application."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

from ai_targeting import (
    AimSettings,
    aim_settings_from_mapping,
    aim_settings_to_mapping,
)
from motion import (
    MOTION_PRESETS,
    MotionSettings,
    motion_settings_from_mapping,
    motion_settings_to_mapping,
)


SCHEMA_VERSION = 3
VALID_BUTTONS = ("Left", "Right", "Middle", "Mouse4", "Mouse5")
VALID_MODES = ("jitter", "ai_aim")
VALID_THEMES = ("light", "dark")


@dataclass(frozen=True)
class AppConfig:
    motion: MotionSettings = field(default_factory=MotionSettings)
    ai: AimSettings = field(default_factory=AimSettings)
    mode: str = "jitter"
    trigger: str = "Left"
    modifier: str = "None"
    hotkey_vk: int = 0xBD
    hotkey_name: str = "-"
    # The motion defaults exactly match the Balanced preset; Custom is only
    # used when the current values do not match a named preset.
    selected_preset: str = "Balanced"
    theme: str = "light"


@dataclass(frozen=True)
class LoadOutcome:
    config: AppConfig
    save_allowed: bool = True
    warning: str | None = None


def runtime_base_dir() -> Path:
    """Return the directory that should hold this app's user data.

    Nuitka exposes the directory of its containing binary through the
    ``__compiled__`` marker.  If that marker is incomplete, the original
    executable argument is the compiled-mode fallback.  Source runs always use
    the directory containing this module.  No EverFall paths are consulted.
    """

    source_dir = Path(__file__).resolve().parent
    try:
        compiled = __compiled__
    except NameError:
        return source_dir

    try:
        containing_dir = getattr(compiled, "containing_dir")
        if not os.fspath(containing_dir):
            raise ValueError
        return Path(containing_dir).resolve()
    except (AttributeError, TypeError, ValueError, OSError, RuntimeError):
        pass

    try:
        executable = sys.argv[0]
        if not os.fspath(executable):
            raise ValueError
        return Path(executable).resolve().parent
    except (IndexError, TypeError, ValueError, OSError, RuntimeError):
        return source_dir


def _safe_int(raw: Any, default: int, low: int, high: int) -> int:
    try:
        # Avoid accepting fractional values as a surprising truncated key.
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        value = default
    return max(low, min(high, value))


def _schema_identifier(raw: Any) -> int:
    """Return an exact integral schema identifier without truncation."""
    if isinstance(raw, bool):
        raise ValueError
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not raw.is_integer():
            raise ValueError
        return int(raw)
    if isinstance(raw, str):
        return int(raw)
    raise TypeError


class ConfigStore:
    """Load and atomically save one schema-versioned Jitter config file."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else runtime_base_dir() / "config.json"
        self._save_allowed = True

    @staticmethod
    def _defaults() -> AppConfig:
        return AppConfig()

    def load(self) -> LoadOutcome:
        self._save_allowed = True
        if not self.path.exists():
            return LoadOutcome(self._defaults())
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, ValueError, UnicodeError) as exc:
            return LoadOutcome(self._defaults(), warning=f"Configuration load failed: {type(exc).__name__}")

        if not isinstance(document, dict):
            return LoadOutcome(self._defaults(), warning="Configuration root must be an object")
        try:
            schema = _schema_identifier(
                document.get("schema_version", SCHEMA_VERSION)
            )
        except (TypeError, ValueError, OverflowError):
            self._save_allowed = False
            return LoadOutcome(
                self._defaults(),
                save_allowed=False,
                warning="Invalid configuration schema",
            )
        if schema > SCHEMA_VERSION:
            self._save_allowed = False
            return LoadOutcome(
                self._defaults(), save_allowed=False,
                warning=f"Unsupported configuration schema: {schema}",
            )
        if schema < 1:
            return LoadOutcome(self._defaults(), warning="Unsupported configuration schema")

        trigger = document.get("trigger", "Left")
        if trigger not in VALID_BUTTONS:
            trigger = "Left"
        modifier = document.get("modifier", "None")
        if modifier != "None" and modifier not in VALID_BUTTONS:
            modifier = "None"
        hotkey_name = document.get("hotkey_name", "-")
        if not isinstance(hotkey_name, str):
            hotkey_name = "-"
        theme = document.get("theme", "light")
        if theme not in VALID_THEMES:
            theme = "light"
        if schema in (1, 2):
            mode = "jitter"
            ai = AimSettings()
        else:
            mode = document.get("mode", "jitter")
            if mode not in VALID_MODES:
                mode = "jitter"
            ai_raw = document.get("ai")
            ai = aim_settings_from_mapping(
                ai_raw if isinstance(ai_raw, Mapping) else None
            )
        if schema == 1:
            motion = MotionSettings()
            selected_preset = "Balanced"
        else:
            motion_raw = document.get("motion")
            motion = motion_settings_from_mapping(
                motion_raw if isinstance(motion_raw, Mapping) else None
            )
            selected_preset = document.get("selected_preset", "Custom")
            if not isinstance(selected_preset, str) or selected_preset not in {"Custom", *MOTION_PRESETS}:
                selected_preset = "Custom"
        config = AppConfig(
            motion=motion,
            ai=ai,
            mode=mode,
            trigger=trigger,
            modifier=modifier,
            hotkey_vk=_safe_int(document.get("hotkey_vk", 0xBD), 0xBD, 1, 255),
            hotkey_name=hotkey_name,
            selected_preset=selected_preset,
            theme=theme,
        )
        return LoadOutcome(config)

    def save(self, config: AppConfig) -> None:
        if not self._save_allowed:
            raise PermissionError("Saving is disabled for an unsupported future configuration schema")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": SCHEMA_VERSION,
            "motion": motion_settings_to_mapping(config.motion),
            "ai": aim_settings_to_mapping(config.ai),
            "mode": config.mode if config.mode in VALID_MODES else "jitter",
            "trigger": config.trigger if config.trigger in VALID_BUTTONS else "Left",
            "modifier": config.modifier if config.modifier == "None" or config.modifier in VALID_BUTTONS else "None",
            "hotkey_vk": _safe_int(config.hotkey_vk, 0xBD, 1, 255),
            "hotkey_name": config.hotkey_name if isinstance(config.hotkey_name, str) else "-",
            "selected_preset": config.selected_preset if isinstance(config.selected_preset, str) and config.selected_preset else "Custom",
            "theme": config.theme if config.theme in VALID_THEMES else "light",
        }
        temporary = Path(str(self.path) + ".tmp")
        backup = Path(str(self.path) + ".bak")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(document, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if self.path.exists():
                shutil.copy2(self.path, backup)
            os.replace(temporary, self.path)
        except OSError:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
