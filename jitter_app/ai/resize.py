"""Pure deterministic NumPy RGB resize shared by AI detection and zoom."""

from functools import lru_cache

import numpy as np


_SEPARABLE_RESIZE_SHAPES = {
    (160, 160, 320),
    (213, 213, 320),
    (320, 320, 160),
    (320, 320, 640),
}


@lru_cache(maxsize=16)
def _resize_plan(
    source_height: int,
    source_width: int,
    output_height: int,
    output_width: int | None = None,
) -> tuple[np.ndarray, ...]:
    if output_width is None:
        output_width = output_height
    source_x = np.linspace(0.0, source_width - 1, output_width)
    source_y = np.linspace(0.0, source_height - 1, output_height)
    x0 = np.floor(source_x).astype(np.intp)
    y0 = np.floor(source_y).astype(np.intp)
    values = (
        x0,
        np.minimum(x0 + 1, source_width - 1),
        y0,
        np.minimum(y0 + 1, source_height - 1),
        (source_x - x0)[None, :, None],
        (source_y - y0)[:, None, None],
    )
    return tuple(
        np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
        for value in values
    )


def _validate_source(image: np.ndarray) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] < 1
        or image.shape[1] < 1
        or image.dtype != np.uint8
    ):
        raise ValueError("Resize source must be a nonempty RGB uint8 array")


def _positive_output_dimension(value: int) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Output size must be a positive integer") from exc
    if isinstance(value, bool) or converted != value or converted < 1:
        raise ValueError("Output size must be a positive integer")
    return converted


def _blend_rgb(
    image: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    output_height, output_width = y0.size, x0.size
    if (
        output_width == output_height
        and (source_height, source_width, output_height) in _SEPARABLE_RESIZE_SHAPES
    ):
        source = image.astype(np.float64)
        horizontal = source[:, x0, :] * (1.0 - wx) + source[:, x1, :] * wx
        blended = horizontal[y0, :, :] * (1.0 - wy) + horizontal[y1, :, :] * wy
    else:
        top_left = image[y0[:, None], x0[None, :]].astype(np.float32)
        top_right = image[y0[:, None], x1[None, :]].astype(np.float32)
        bottom_left = image[y1[:, None], x0[None, :]].astype(np.float32)
        bottom_right = image[y1[:, None], x1[None, :]].astype(np.float32)
        blended = (
            top_left * (1.0 - wx) * (1.0 - wy)
            + top_right * wx * (1.0 - wy)
            + bottom_left * (1.0 - wx) * wy
            + bottom_right * wx * wy
        )
    rounded = np.floor(np.clip(blended, 0.0, 255.0) + 0.5)
    return np.ascontiguousarray(rounded.astype(np.uint8))


def resize_rgb_bilinear_to(
    image: np.ndarray,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    _validate_source(image)
    output_width = _positive_output_dimension(output_width)
    output_height = _positive_output_dimension(output_height)
    source_height, source_width = image.shape[:2]
    x0, x1, y0, y1, wx, wy = _resize_plan(
        source_height,
        source_width,
        output_height,
        output_width,
    )
    return _blend_rgb(image, x0, x1, y0, y1, wx, wy)


def resize_rgb_bilinear(
    image: np.ndarray,
    output_size: int = 320,
) -> np.ndarray:
    return resize_rgb_bilinear_to(image, output_size, output_size)
