"""Pure NumPy post-processing for exact raw one-class YOLO outputs."""

import numpy as np

from .targeting import Detection


LOGICAL_FRAME_SIZE = 320
MIN_CONFIDENCE = 0.05
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 300
RAW_CANDIDATE_COUNTS = {160: 525, 320: 2100, 640: 8400}


def _nms_keep_positions(
    boxes: np.ndarray,
    confidences: np.ndarray,
    raw_indices: np.ndarray,
) -> np.ndarray:
    order = np.argsort(-confidences, kind="stable")
    kept: list[int] = []
    while order.size and len(kept) < MAX_DETECTIONS:
        current = int(order[0])
        kept.append(current)
        remaining = order[1:]
        if not remaining.size:
            break
        others = boxes[remaining]
        left = np.maximum(boxes[current, 0], others[:, 0])
        top = np.maximum(boxes[current, 1], others[:, 1])
        right = np.minimum(boxes[current, 2], others[:, 2])
        bottom = np.minimum(boxes[current, 3], others[:, 3])
        intersection = np.maximum(0.0, right - left) * np.maximum(
            0.0, bottom - top
        )
        current_area = (
            (boxes[current, 2] - boxes[current, 0])
            * (boxes[current, 3] - boxes[current, 1])
        )
        other_areas = (
            (others[:, 2] - others[:, 0])
            * (others[:, 3] - others[:, 1])
        )
        union = current_area + other_areas - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        )
        order = remaining[iou <= NMS_IOU_THRESHOLD]
    kept_array = np.asarray(kept, dtype=np.int64)
    if not kept_array.size:
        return kept_array
    return kept_array[np.argsort(raw_indices[kept_array], kind="stable")]


def decode_single_class_yolo(
    output: np.ndarray,
    input_size: int,
) -> tuple[Detection, ...]:
    if type(input_size) is not int:
        raise ValueError("Raw YOLO input size must be 160, 320, or 640")
    expected_candidates = RAW_CANDIDATE_COUNTS.get(input_size)
    expected_shape = (1, 5, expected_candidates) if expected_candidates else None
    if (
        not isinstance(output, np.ndarray)
        or output.shape != expected_shape
        or not np.isrealobj(output)
        or not (
            np.issubdtype(output.dtype, np.integer)
            or np.issubdtype(output.dtype, np.floating)
        )
    ):
        raise ValueError(
            "Raw YOLO output must be a numeric [1,5,K] tensor matching "
            "input size 160, 320, or 640"
        )

    rows = output[0].T.astype(np.float64, copy=False)
    finite = np.isfinite(rows).all(axis=1)
    confidence = rows[:, 4]
    valid = (
        finite
        & (confidence >= MIN_CONFIDENCE)
        & (confidence <= 1.0)
        & (rows[:, 2] > 0.0)
        & (rows[:, 3] > 0.0)
    )
    raw_indices = np.flatnonzero(valid)
    if not raw_indices.size:
        return ()
    rows = rows[raw_indices]
    boxes = np.column_stack((
        rows[:, 0] - rows[:, 2] / 2.0,
        rows[:, 1] - rows[:, 3] / 2.0,
        rows[:, 0] + rows[:, 2] / 2.0,
        rows[:, 1] + rows[:, 3] / 2.0,
    ))
    finite_boxes = np.isfinite(boxes).all(axis=1)
    boxes = boxes[finite_boxes]
    confidences = rows[finite_boxes, 4]
    raw_indices = raw_indices[finite_boxes]
    if not raw_indices.size:
        return ()
    kept = _nms_keep_positions(boxes, confidences, raw_indices)
    detections = []
    for position in kept:
        x1, y1, x2, y2 = np.clip(boxes[position], 0.0, input_size)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(
            float(x1), float(y1), float(x2), float(y2),
            float(confidences[position]), 0,
        ))
    return tuple(detections)
