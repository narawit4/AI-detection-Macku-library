"""Resolve Jitter resources without depending on the working directory."""

from pathlib import Path


def bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bundled_model_path() -> Path:
    return bundle_root() / "models" / "all_games_320.onnx"


def sound_directory() -> Path:
    return bundle_root() / "sound"
