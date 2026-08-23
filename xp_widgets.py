from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import math
import tkinter as tk
from typing import Callable


class XPGlossySlider(tk.Canvas):
    THUMB_RADIUS = 8
    RAIL_INSET = 14

    def __init__(
        self,
        parent,
        *,
        from_: float,
        to: float,
        resolution: float,
        command: Callable[[str], None] | None = None,
        width: int = 220,
        height: int = 34,
    ) -> None:
        if not math.isfinite(float(from_)) or not math.isfinite(float(to)):
            raise ValueError("slider range must be finite")
        if float(to) <= float(from_):
            raise ValueError("slider maximum must be greater than minimum")
        if not math.isfinite(float(resolution)) or float(resolution) <= 0:
            raise ValueError("slider resolution must be positive and finite")
        super().__init__(
            parent,
            width=width,
            height=height,
            highlightthickness=0,
            borderwidth=0,
            takefocus=True,
            background="#F4F1E6",
        )
        self.from_ = float(from_)
        self.to = float(to)
        self.resolution = float(resolution)
        self.command = command
        self._value = self.from_
        self._destroyed = False
        self._bubble_after_id: str | None = None
        self.bind("<Configure>", self._on_configure)
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _decimal_places(self) -> int:
        return max(
            0,
            -Decimal(str(self.resolution)).normalize().as_tuple().exponent,
        )

    def _snap(self, value: float) -> float:
        numeric = min(self.to, max(self.from_, float(value)))
        origin = Decimal(str(self.from_))
        step = Decimal(str(self.resolution))
        count = ((Decimal(str(numeric)) - origin) / step).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
        snapped = origin + count * step
        clamped = min(Decimal(str(self.to)), max(origin, snapped))
        return float(clamped)

    def _format_value(self, value: float) -> str:
        places = self._decimal_places()
        if places == 0:
            return str(int(round(value)))
        return f"{value:.{places}f}".rstrip("0").rstrip(".")

    def get(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._value = self._snap(value)
        self._redraw()

    def _rail_bounds(self) -> tuple[float, float]:
        width = max(self.winfo_width(), self.winfo_reqwidth())
        return (
            float(self.RAIL_INSET),
            float(max(self.RAIL_INSET, width - self.RAIL_INSET)),
        )

    def _value_to_x(self, value: float) -> float:
        left, right = self._rail_bounds()
        ratio = (self._snap(value) - self.from_) / (self.to - self.from_)
        return left + ratio * (right - left)

    def _x_to_value(self, x: float) -> float:
        left, right = self._rail_bounds()
        if right <= left:
            return self.from_
        ratio = min(1.0, max(0.0, (float(x) - left) / (right - left)))
        return self._snap(self.from_ + ratio * (self.to - self.from_))

    def _redraw(self) -> None:
        pass

    def _on_configure(self, _event=None) -> None:
        self._redraw()

    def _on_destroy(self, _event=None) -> None:
        self._destroyed = True
        if self._bubble_after_id is not None:
            try:
                self.after_cancel(self._bubble_after_id)
            except tk.TclError:
                pass
            self._bubble_after_id = None
