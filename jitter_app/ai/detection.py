"""Exact dual-contract ONNX detection for AI aim models."""

import ast
from dataclasses import dataclass
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort

from .targeting import Detection
from .yolo import RAW_CANDIDATE_COUNTS, decode_single_class_yolo
from .resize import resize_rgb_bilinear_to
from jitter_app.resources import bundled_model_path


LOGICAL_FRAME_SIZE = 320
SUPPORTED_INPUT_SIZES: tuple[int, ...] = (160, 320, 640)
_INPUT_NAME = "images"
_OUTPUT_NAME = "output0"
_POST_NMS_OUTPUT_SHAPE = [1, 300, 6]
_POST_NMS_FORMAT = "post_nms"
_RAW_SINGLE_CLASS_FORMAT = "raw_single_class"
_TENSOR_TYPE = "tensor(float)"
_OUTPUT_CONTRACT_MESSAGE = (
    "AI model output must be output0 tensor(float) [1,300,6] "
    "or supported raw one-class [1,5,K]"
)
_RAW_METADATA_MESSAGE = (
    "Raw YOLO metadata must declare task=detect and exactly one named class 0"
)


class ModelContractError(RuntimeError):
    """Raised when a model or inference result differs from the fixed contract."""


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    input_size: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


def _validate_raw_metadata(session: object) -> None:
    try:
        metadata = session.get_modelmeta().custom_metadata_map
    except Exception as error:
        raise ModelContractError(_RAW_METADATA_MESSAGE) from error
    if (
        not isinstance(metadata, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
        or metadata.get("task") != "detect"
    ):
        raise ModelContractError(_RAW_METADATA_MESSAGE)
    raw_names = metadata.get("names")
    if not isinstance(raw_names, str):
        raise ModelContractError(_RAW_METADATA_MESSAGE)
    try:
        names = ast.literal_eval(raw_names)
    except (SyntaxError, ValueError, TypeError) as error:
        raise ModelContractError(_RAW_METADATA_MESSAGE) from error
    if not isinstance(names, dict) or len(names) != 1:
        raise ModelContractError(_RAW_METADATA_MESSAGE)
    key, label = next(iter(names.items()))
    if type(key) is not int or key != 0 or not isinstance(label, str) or not label:
        raise ModelContractError(_RAW_METADATA_MESSAGE)


def model_resource_path() -> Path:
    return bundled_model_path()


def _validated_input_size(raw: object) -> int:
    if type(raw) is not int or raw not in SUPPORTED_INPUT_SIZES:
        raise ModelContractError(
            "AI model input must be images tensor(float) "
            "[1,3,N,N] where N is 160, 320, or 640"
        )
    return raw


def build_letterbox_transform(
    source_width: int,
    source_height: int,
    input_size: int = LOGICAL_FRAME_SIZE,
) -> LetterboxTransform:
    input_size = _validated_input_size(input_size)
    if (
        type(source_width) is not int
        or type(source_height) is not int
        or source_width <= 0
        or source_height <= 0
    ):
        raise ValueError("AI frame dimensions must be positive integers")
    gain = min(input_size / source_width, input_size / source_height)
    resized_width = min(
        input_size, max(1, math.floor(source_width * gain + 0.5))
    )
    resized_height = min(
        input_size, max(1, math.floor(source_height * gain + 0.5))
    )
    horizontal = input_size - resized_width
    vertical = input_size - resized_height
    left = horizontal // 2
    top = vertical // 2
    return LetterboxTransform(
        source_width,
        source_height,
        input_size,
        resized_width,
        resized_height,
        left,
        top,
        horizontal - left,
        vertical - top,
    )


def _prepare_frame(
    frame: np.ndarray,
    input_size: int = LOGICAL_FRAME_SIZE,
) -> tuple[np.ndarray, LetterboxTransform]:
    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError("AI frame must be RGB with three channels")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError("AI frame dimensions must be positive")
    if frame.dtype != np.uint8:
        raise ValueError("AI frame must use uint8 pixels")
    transform = build_letterbox_transform(
        frame.shape[1], frame.shape[0], input_size
    )
    content = resize_rgb_bilinear_to(
        frame, transform.resized_width, transform.resized_height
    )
    prepared = np.full(
        (transform.input_size, transform.input_size, 3), 114, dtype=np.uint8
    )
    prepared[
        transform.pad_top:transform.pad_top + transform.resized_height,
        transform.pad_left:transform.pad_left + transform.resized_width,
    ] = content
    tensor = np.ascontiguousarray(
        prepared.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    )
    return tensor, transform


def preprocess_frame(
    frame: np.ndarray,
    input_size: int = LOGICAL_FRAME_SIZE,
) -> np.ndarray:
    tensor, _ = _prepare_frame(frame, input_size)
    return tensor


def map_detection_to_source(
    detection: Detection,
    transform: LetterboxTransform,
) -> Detection | None:
    content_right = transform.pad_left + transform.resized_width
    content_bottom = transform.pad_top + transform.resized_height
    x1 = max(float(transform.pad_left), detection.x1)
    y1 = max(float(transform.pad_top), detection.y1)
    x2 = min(float(content_right), detection.x2)
    y2 = min(float(content_bottom), detection.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    x_scale = transform.source_width / transform.resized_width
    y_scale = transform.source_height / transform.resized_height
    return Detection(
        max(0.0, min(transform.source_width,
                     (x1 - transform.pad_left) * x_scale)),
        max(0.0, min(transform.source_height,
                     (y1 - transform.pad_top) * y_scale)),
        max(0.0, min(transform.source_width,
                     (x2 - transform.pad_left) * x_scale)),
        max(0.0, min(transform.source_height,
                     (y2 - transform.pad_top) * y_scale)),
        detection.confidence,
        detection.class_id,
    )


def parse_output(
    output: np.ndarray,
    input_size: int = LOGICAL_FRAME_SIZE,
) -> tuple[Detection, ...]:
    input_size = _validated_input_size(input_size)
    if (
        not isinstance(output, np.ndarray)
        or output.shape != tuple(_POST_NMS_OUTPUT_SHAPE)
    ):
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
        x1 = min(input_size, max(0.0, x1))
        y1 = min(input_size, max(0.0, y1))
        x2 = min(input_size, max(0.0, x2))
        y2 = min(input_size, max(0.0, y2))
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
        self._input_size, self._output_format = self._validate_contract()
        providers = self._session.get_providers()
        if not providers:
            raise ModelContractError("AI model session has no execution provider")
        self.provider = providers[0]

    @property
    def input_size(self) -> int:
        return self._input_size

    def _validate_contract(self) -> tuple[int, str]:
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ModelContractError("AI model must have exactly one input")
        node = inputs[0]
        shape = list(node.shape)
        if (
            node.name != _INPUT_NAME
            or node.type != _TENSOR_TYPE
            or len(shape) != 4
            or any(type(dimension) is not int for dimension in shape)
            or shape[:2] != [1, 3]
            or shape[2] != shape[3]
        ):
            raise ModelContractError(
                "AI model input must be images tensor(float) "
                "[1,3,N,N] where N is 160, 320, or 640"
            )
        input_size = _validated_input_size(shape[2])
        return input_size, self._validate_output_contract(input_size)

    def _validate_output_contract(self, input_size: int) -> str:
        outputs = self._session.get_outputs()
        if len(outputs) != 1:
            raise ModelContractError("AI model must have exactly one output")
        node = outputs[0]
        shape = list(node.shape)
        if (
            node.name != _OUTPUT_NAME
            or node.type != _TENSOR_TYPE
            or any(type(dimension) is not int for dimension in shape)
        ):
            raise ModelContractError(_OUTPUT_CONTRACT_MESSAGE)
        if shape == _POST_NMS_OUTPUT_SHAPE:
            return _POST_NMS_FORMAT
        if shape == [1, 5, RAW_CANDIDATE_COUNTS[input_size]]:
            _validate_raw_metadata(self._session)
            return _RAW_SINGLE_CLASS_FORMAT
        raise ModelContractError(_OUTPUT_CONTRACT_MESSAGE)

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        tensor, transform = _prepare_frame(frame, self._input_size)
        output = self._session.run([_OUTPUT_NAME], {_INPUT_NAME: tensor})[0]
        try:
            if self._output_format == _POST_NMS_FORMAT:
                detections = parse_output(output, self._input_size)
            else:
                detections = decode_single_class_yolo(output, self._input_size)
            return tuple(
                mapped
                for detection in detections
                if (mapped := map_detection_to_source(detection, transform))
                is not None
            )
        except (ModelContractError, TypeError, ValueError) as error:
            raise ModelContractError(_OUTPUT_CONTRACT_MESSAGE) from error
