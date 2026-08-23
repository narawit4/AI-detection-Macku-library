"""Windows process entry point for the standalone Jitter application."""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import Any

from settings import ConfigStore, runtime_base_dir
from ui import JitterApp


MUTEX_NAME = r"Local\Jitter_Makcu_Controller_Mutex"
ERROR_ALREADY_EXISTS = 183

_mutex_handle: Any | None = None


def ensure_single_instance(kernel32: Any | None = None) -> Any | None:
    """Acquire the process mutex, returning its handle for the first instance.

    ``kernel32`` is injectable so this Windows-only boundary can be tested
    without creating a real system mutex.  The first handle is retained in a
    module global for the lifetime of the process.
    """

    global _mutex_handle
    api = kernel32 or ctypes.windll.kernel32
    handle = api.CreateMutexW(None, False, MUTEX_NAME)
    if not handle:
        return None
    if api.GetLastError() == ERROR_ALREADY_EXISTS:
        api.CloseHandle(handle)
        return None
    _mutex_handle = handle
    return handle


def configure_logging(base_dir: Path) -> None:
    """Configure a quiet timestamped file logger in the application folder."""

    base_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / "app.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicate lines when an embedding host calls main more than once,
    # while leaving unrelated handlers intact.
    for handler in list(root_logger.handlers):
        if getattr(handler, "_jitter_log_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler._jitter_log_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger.addHandler(handler)


def _show_duplicate_message() -> None:
    try:
        ctypes.windll.user32.MessageBoxW(
            0,
            "Jitter is already running.",
            "Jitter",
            0x00000010,
        )
    except (AttributeError, OSError):
        logging.error("A second Jitter instance was requested.")


def main() -> None:
    """Start Jitter from source or a packaged executable."""

    base_dir = runtime_base_dir()
    configure_logging(base_dir)
    logging.info("Starting Jitter from %s", base_dir)
    if ensure_single_instance() is None:
        _show_duplicate_message()
        return

    config_store = ConfigStore(base_dir / "config.json")
    app = JitterApp(config_store=config_store)
    try:
        app.mainloop()
    finally:
        logging.info("Jitter stopped")


if __name__ == "__main__":
    main()
