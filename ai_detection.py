"""Fixed-contract ONNX detection for the bundled AI aim model."""

from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort

from ai_targeting import Detection


_FRAME_SIZE = 320
_INPUT_NAME = "images"
_OUTPUT_NAME = "output0"
_INPUT_SHAPE = [1, 3, _FRAME_SIZE, _FRAME_SIZE]
_OUTPUT_SHAPE = [1, 300, 6]
_TENSOR_TYPE = "tensor(float)"


class ModelContractError(RuntimeError):
    """Raised when a model or inference result differs from the fixed contract."""


def model_resource_path() -> Path:
    return Path(__file__).resolve().parent / "models" / "all_games_320.onnx"


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.shape != (_FRAME_SIZE, _FRAME_SIZE, 3):
        raise ValueError("AI frame must be RGB 320x320x3")
    if frame.dtype != np.uint8:
        raise ValueError("AI frame must use uint8 pixels")
    return np.ascontiguousarray(
        frame.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    )


def parse_output(output: np.ndarray) -> tuple[Detection, ...]:
    if not isinstance(output, np.ndarray) or output.shape != tuple(_OUTPUT_SHAPE):
        raise ModelContractError(
            "AI model output must have shape [1, 300, 6]"
        )
    try:
        finite_rows = np.isfinite(output).all(axis=2)[0]
    except TypeError as error:
        raise ModelContractError("AI model output must contain numeric values") from error

    detections = []
    for row in output[0, finite_rows]:
        x1, y1, x2, y2, confidence, class_id = (float(value) for value in row)
        if not 0.0 <= confidence <= 1.0:
            continue
        if class_id < 0.0 or not class_id.is_integer():
            continue
        x1 = min(_FRAME_SIZE, max(0.0, x1))
        y1 = min(_FRAME_SIZE, max(0.0, y1))
        x2 = min(_FRAME_SIZE, max(0.0, x2))
        y2 = min(_FRAME_SIZE, max(0.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(x1, y1, x2, y2, confidence, int(class_id)))
    return tuple(detections)


class OnnxDetector:
    """Inference wrapper that accepts only the approved model contract."""

    def __init__(
        self,
        model_path: Path | str,
        session_factory: Callable | None = None,
    ) -> None:
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        factory = session_factory or ort.InferenceSession
        try:
            self._session = factory(
                str(model_path),
                sess_options=options,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception:
            self._session = factory(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        self._validate_contract()
        providers = self._session.get_providers()
        if not providers:
            raise ModelContractError("AI model session has no execution provider")
        self.provider = providers[0]

    def _validate_contract(self) -> None:
        self._validate_node(
            self._session.get_inputs(), _INPUT_NAME, _INPUT_SHAPE, "input"
        )
        self._validate_node(
            self._session.get_outputs(), _OUTPUT_NAME, _OUTPUT_SHAPE, "output"
        )

    @staticmethod
    def _validate_node(nodes, name: str, shape: list[int], kind: str) -> None:
        if len(nodes) != 1:
            raise ModelContractError(f"AI model must have exactly one {kind}")
        node = nodes[0]
        if (
            node.name != name
            or node.type != _TENSOR_TYPE
            or list(node.shape) != shape
        ):
            raise ModelContractError(
                f"AI model {kind} must be {name} {_TENSOR_TYPE}{shape}"
            )

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        tensor = preprocess_frame(frame)
        output = self._session.run([_OUTPUT_NAME], {_INPUT_NAME: tensor})[0]
        return parse_output(output)
