"""Primary-display refresh detection and runtime cadence policy."""

from __future__ import annotations

import ctypes
import logging
import math
from dataclasses import dataclass
from typing import Any
from ctypes import wintypes


@dataclass(frozen=True)
class RuntimeCadence:
    display_hz: int | None
    capture_fps: int
    servo_hz: int


FALLBACK_CADENCE = RuntimeCadence(None, 120, 240)
ENUM_CURRENT_SETTINGS = -1


class _POINTL(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _PRINTER_FIELDS(ctypes.Structure):
    _fields_ = [
        ("dmOrientation", wintypes.SHORT), ("dmPaperSize", wintypes.SHORT),
        ("dmPaperLength", wintypes.SHORT), ("dmPaperWidth", wintypes.SHORT),
        ("dmScale", wintypes.SHORT), ("dmCopies", wintypes.SHORT),
        ("dmDefaultSource", wintypes.SHORT), ("dmPrintQuality", wintypes.SHORT),
    ]


class _DISPLAY_FIELDS(ctypes.Structure):
    _fields_ = [
        ("dmPosition", _POINTL),
        ("dmDisplayOrientation", wintypes.DWORD),
        ("dmDisplayFixedOutput", wintypes.DWORD),
    ]


class _MODE_FIELDS(ctypes.Union):
    _fields_ = [("printer", _PRINTER_FIELDS), ("display", _DISPLAY_FIELDS)]


class _DISPLAY_FLAGS(ctypes.Union):
    _fields_ = [("dmDisplayFlags", wintypes.DWORD), ("dmNup", wintypes.DWORD)]


class _DEVMODEW(ctypes.Structure):
    _anonymous_ = ("mode", "flags")
    _fields_ = [
        ("dmDeviceName", wintypes.WCHAR * 32),
        ("dmSpecVersion", wintypes.WORD), ("dmDriverVersion", wintypes.WORD),
        ("dmSize", wintypes.WORD), ("dmDriverExtra", wintypes.WORD),
        ("dmFields", wintypes.DWORD), ("mode", _MODE_FIELDS),
        ("dmColor", wintypes.SHORT), ("dmDuplex", wintypes.SHORT),
        ("dmYResolution", wintypes.SHORT), ("dmTTOption", wintypes.SHORT),
        ("dmCollate", wintypes.SHORT), ("dmFormName", wintypes.WCHAR * 32),
        ("dmLogPixels", wintypes.WORD), ("dmBitsPerPel", wintypes.DWORD),
        ("dmPelsWidth", wintypes.DWORD), ("dmPelsHeight", wintypes.DWORD),
        ("flags", _DISPLAY_FLAGS), ("dmDisplayFrequency", wintypes.DWORD),
        ("dmICMMethod", wintypes.DWORD), ("dmICMIntent", wintypes.DWORD),
        ("dmMediaType", wintypes.DWORD), ("dmDitherType", wintypes.DWORD),
        ("dmReserved1", wintypes.DWORD), ("dmReserved2", wintypes.DWORD),
        ("dmPanningWidth", wintypes.DWORD), ("dmPanningHeight", wintypes.DWORD),
    ]


def cadence_from_refresh(raw: Any) -> RuntimeCadence:
    try:
        if isinstance(raw, bool):
            raise TypeError
        value = float(raw)
        if not math.isfinite(value) or not 24.0 <= value <= 500.0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        return FALLBACK_CADENCE
    display_hz = int(round(value))
    return RuntimeCadence(
        display_hz,
        min(display_hz, 240),
        max(120, min(480, display_hz * 2)),
    )


def detect_runtime_cadence(user32: Any | None = None) -> RuntimeCadence:
    try:
        if user32 is None:
            user32 = ctypes.windll.user32
        mode = _DEVMODEW()
        mode.dmSize = ctypes.sizeof(_DEVMODEW)
        if not user32.EnumDisplaySettingsW(None, ENUM_CURRENT_SETTINGS, ctypes.byref(mode)):
            return FALLBACK_CADENCE
        return cadence_from_refresh(mode.dmDisplayFrequency)
    except Exception:
        logging.exception("Unable to detect primary display refresh rate")
        return FALLBACK_CADENCE
