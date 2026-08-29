"""Windows process entry point for the standalone Jitter application."""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Callable, TextIO

MUTEX_NAME = r"Local\Jitter_Makcu_Controller_Mutex"
ERROR_ALREADY_EXISTS = 183
AI_RUNTIME_SELF_CHECK_ARGUMENT = "--ai-runtime-self-check"
AI_MODEL_SHA256 = "6B9157D6419F9DBC40D2DCECCC33A3387078C86F1C5872EDA544B174FF48499C"
REQUIRED_AI_PROVIDER = "DmlExecutionProvider"

_mutex_handle: Any | None = None


class MutexCreationError(RuntimeError):
    """Raised when Windows cannot create the single-instance mutex."""

    def __init__(self, error_code: int) -> None:
        self.error_code = int(error_code)
        super().__init__(f"CreateMutexW failed with Windows error {self.error_code}")


def runtime_base_dir() -> Path:
    """Resolve the normal UI data directory without importing it for self-checks."""
    from jitter_app.config.store import runtime_base_dir as resolve_runtime_base_dir

    return resolve_runtime_base_dir()


def ConfigStore(*args: Any, **kwargs: Any) -> Any:
    """Construct the normal configuration store lazily."""
    from jitter_app.config.store import ConfigStore as ConfigStoreType

    return ConfigStoreType(*args, **kwargs)


def JitterApp(*args: Any, **kwargs: Any) -> Any:
    """Construct the Tk application lazily so self-checks never import Tk."""
    from jitter_app.presentation.ui import JitterApp as JitterAppType

    return JitterAppType(*args, **kwargs)


def run_ai_runtime_self_check(
    *,
    model_path: Path | str | None = None,
    detector_factory: Callable[[Path], Any] | None = None,
    output: TextIO | None = None,
) -> int:
    """Validate the bundled model contract and require DirectML, reporting JSON."""
    payload: dict[str, object] = {
        "status": "error",
        "required_provider": REQUIRED_AI_PROVIDER,
        "provider": None,
        "model_path": None,
        "model_sha256": None,
        "expected_model_sha256": AI_MODEL_SHA256,
    }
    result = 1
    try:
        factory = detector_factory
        resource_path = None
        if factory is None or model_path is None:
            from jitter_app.ai.detection import OnnxDetector, model_resource_path

            factory = factory or OnnxDetector
            resource_path = model_resource_path

        model = Path(model_path) if model_path is not None else resource_path()
        model = model.resolve()
        payload["model_path"] = str(model)
        model_hash = hashlib.sha256(model.read_bytes()).hexdigest().upper()
        payload["model_sha256"] = model_hash
        if model_hash != AI_MODEL_SHA256:
            payload["error"] = "model_hash_mismatch"
        else:
            detector = factory(model)
            provider = detector.provider
            if isinstance(provider, str):
                payload["provider"] = provider
            if provider != REQUIRED_AI_PROVIDER:
                payload["error"] = "required_provider_unavailable"
            else:
                payload["status"] = "ok"
                result = 0
    except Exception:
        payload["error"] = "runtime_initialization_failed"

    stream = output if output is not None else (sys.stdout or sys.stderr)
    if stream is not None:
        print(json.dumps(payload, sort_keys=True), file=stream)
    return result


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
        raise MutexCreationError(api.GetLastError())
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


def _show_startup_error(message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(0, message, "Jitter startup error", 0x00000010)
    except (AttributeError, OSError):
        logging.error(message)


def main() -> None:
    """Start Jitter from source or a packaged executable."""

    base_dir = runtime_base_dir()
    configure_logging(base_dir)
    logging.info("Starting Jitter from %s", base_dir)
    try:
        first_handle = ensure_single_instance()
    except MutexCreationError as exc:
        message = f"Unable to create the Jitter single-instance mutex (Windows error {exc.error_code})."
        logging.error(message)
        _show_startup_error(message)
        return
    if first_handle is None:
        _show_duplicate_message()
        return

    config_store = ConfigStore(base_dir / "config.json")
    app = JitterApp(config_store=config_store)
    try:
        app.mainloop()
    finally:
        logging.info("Jitter stopped")


def entrypoint(argv: list[str] | None = None) -> int:
    """Dispatch the exact non-GUI self-check or start the ordinary UI."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == [AI_RUNTIME_SELF_CHECK_ARGUMENT]:
        return run_ai_runtime_self_check()
    if arguments:
        print(
            f"Invalid arguments. Use exactly {AI_RUNTIME_SELF_CHECK_ARGUMENT} or no arguments.",
            file=sys.stderr,
        )
        return 2
    main()
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
