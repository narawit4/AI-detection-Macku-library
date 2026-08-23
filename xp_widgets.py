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
        self._bubble_visible = False
        self._hovered = False
        self._pressed = False
        self._focused = False
        self.bind("<Configure>", self._on_configure)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        for sequence, delta in (
            ("<Left>", -1),
            ("<Down>", -1),
            ("<Right>", 1),
            ("<Up>", 1),
        ):
            self.bind(
                sequence,
                lambda _event, step=delta: self._on_step(step),
            )
        self.bind("<Home>", lambda _event: self._set_from_user(self.from_))
        self.bind("<End>", lambda _event: self._set_from_user(self.to))
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
        if self._destroyed or not self.winfo_exists():
            return
        self.delete("all")
        height = max(self.winfo_height(), self.winfo_reqheight())
        left, right = self._rail_bounds()
        center_y = min(height - 10, 21)
        thumb_x = self._value_to_x(self._value)

        self.create_rectangle(
            left,
            center_y - 3,
            right,
            center_y + 3,
            fill="#E5E2D8",
            outline="#8E9AA6",
            tags="rail",
        )
        fill_width = max(0.0, thumb_x - left)
        colors = (
            "#8EB9E8",
            "#73A7DE",
            "#5B92CC",
            "#356FAF",
            "#2C6099",
            "#244F7D",
        )
        if fill_width > 0:
            segment = fill_width / len(colors)
            for index, color in enumerate(colors):
                x1 = left + index * segment
                x2 = left + (index + 1) * segment
                self.create_rectangle(
                    x1,
                    center_y - 2,
                    x2,
                    center_y + 2,
                    fill=color,
                    outline=color,
                    tags="fill",
                )

        if self._focused:
            self.create_oval(
                thumb_x - 11,
                center_y - 11,
                thumb_x + 11,
                center_y + 11,
                outline="#E4A43A",
                width=2,
                tags="focus",
            )
        if self._hovered and not self._pressed:
            self.create_oval(
                thumb_x - 10,
                center_y - 10,
                thumb_x + 10,
                center_y + 10,
                fill="#D9EAFB",
                outline="",
                tags="halo",
            )

        shadow_y = 2 if not self._pressed else 1
        self.create_oval(
            thumb_x - 8,
            center_y - 6 + shadow_y,
            thumb_x + 8,
            center_y + 10,
            fill="#75828E",
            outline="",
            tags=("thumb", "thumb_shadow"),
        )
        body_fill = "#356FAF" if self._pressed else "#F7F3E7"
        self.create_oval(
            thumb_x - 8,
            center_y - 8,
            thumb_x + 8,
            center_y + 8,
            fill=body_fill,
            outline="#244F7D",
            width=1,
            tags=("thumb", "thumb_body"),
        )
        if not self._pressed:
            self.create_arc(
                thumb_x - 6,
                center_y - 6,
                thumb_x + 6,
                center_y + 5,
                start=20,
                extent=140,
                style="arc",
                outline="#FFFFFF",
                width=2,
                tags=("thumb", "thumb_highlight"),
            )

        if self._bubble_visible or self._hovered or self._pressed:
            text = self._format_value(self._value)
            bubble_y = max(8, center_y - 18)
            half_width = max(13, 4 + len(text) * 4)
            self.create_rectangle(
                thumb_x - half_width,
                bubble_y - 7,
                thumb_x + half_width,
                bubble_y + 7,
                fill="#FFFDF5",
                outline="#356FAF",
                tags="bubble",
            )
            self.create_text(
                thumb_x,
                bubble_y,
                text=text,
                fill="#20252A",
                font=("Tahoma", 8),
                tags="bubble",
            )

    def _set_from_user(self, value: float) -> str:
        self._value = self._snap(value)
        self._show_bubble()
        self._redraw()
        if self.command is not None:
            self.command(self._format_value(self._value))
        return "break"

    def _on_enter(self, _event=None) -> None:
        self._hovered = True
        self._show_bubble()
        self._redraw()

    def _on_leave(self, _event=None) -> None:
        self._hovered = False
        if not self._pressed:
            self._schedule_hide_bubble()
        self._redraw()

    def _on_press(self, event) -> str:
        self.focus_set()
        self._pressed = True
        self.grab_set()
        return self._set_from_user(self._x_to_value(event.x))

    def _on_drag(self, event) -> str:
        return self._set_from_user(self._x_to_value(event.x))

    def _on_release(self, event) -> str:
        self._pressed = False
        try:
            self.grab_release()
        except tk.TclError:
            pass
        result = self._set_from_user(self._x_to_value(event.x))
        if not self._hovered:
            self._schedule_hide_bubble()
        return result

    def _on_step(self, direction: int) -> str:
        return self._set_from_user(self._value + direction * self.resolution)

    def _on_focus_in(self, _event=None) -> None:
        self._focused = True
        self._redraw()

    def _on_focus_out(self, _event=None) -> None:
        self._focused = False
        self._redraw()

    def _show_bubble(self) -> None:
        self._cancel_bubble_hide()
        self._bubble_visible = True

    def _cancel_bubble_hide(self) -> None:
        if self._bubble_after_id is None:
            return
        try:
            self.after_cancel(self._bubble_after_id)
        except tk.TclError:
            pass
        self._bubble_after_id = None

    def _schedule_hide_bubble(self) -> None:
        self._cancel_bubble_hide()
        self._bubble_after_id = self.after(450, self._hide_bubble)

    def _hide_bubble(self) -> None:
        self._bubble_after_id = None
        if self._destroyed:
            return
        self._bubble_visible = False
        self._redraw()

    def _on_configure(self, _event=None) -> None:
        self._redraw()

    def _on_destroy(self, _event=None) -> None:
        self._cancel_bubble_hide()
        self._destroyed = True
