from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import math
import tkinter as tk
from typing import Callable, Mapping


LIGHT_SLIDER_PALETTE = {
    "background": "#F4F1E6", "rail": "#E5E2D8", "rail_outline": "#8E9AA6",
    "hover": "#D9EAFB", "shadow": "#75828E", "thumb": "#F7F3E7",
    "thumb_pressed": "#356FAF", "thumb_outline": "#244F7D",
    "highlight": "#FFFFFF", "bubble": "#FFFDF5", "text": "#20252A",
    "focus": "#E4A43A",
}


LIGHT_NAV_PALETTE = {
    "background": "#F4F1E6",
    "capsule": "#D9EAFB",
    "capsule_outline": "#8E9AA6",
    "pill": "#356FAF",
    "pill_highlight": "#FFFFFF",
    "text": "#20252A",
    "active_text": "#FFFFFF",
    "focus": "#E4A43A",
}


class LiquidXPNav(tk.Canvas):
    """A small, keyboard-accessible tab strip for the Liquid XP layout."""

    def __init__(
        self,
        parent,
        *,
        labels: tuple[str, ...],
        command: Callable[[int], None] | None = None,
        selected: int = 0,
        palette: Mapping[str, str] | None = None,
        animation_ms: int = 160,
        width: int = 330,
        height: int = 38,
    ) -> None:
        self.labels = tuple(labels)
        if not self.labels or any(
            not isinstance(label, str) or not label.strip()
            for label in self.labels
        ):
            raise ValueError("navigation labels must be non-empty")

        self._palette = dict(LIGHT_NAV_PALETTE)
        if palette:
            self._palette.update(palette)
        super().__init__(
            parent,
            width=width,
            height=height,
            highlightthickness=0,
            borderwidth=0,
            takefocus=True,
            background=self._palette["background"],
        )
        self.command = command
        self.animation_ms = int(animation_ms)
        self.selected_index = self._clamp_index(selected)
        self._destroyed = False
        self._focused = False
        self._animation_after_id: str | None = None
        self._pill_x = 0.0
        self._pill_initialized = False

        self.bind("<Configure>", self._on_configure)
        self.bind("<Button-1>", self._on_click)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Left>", lambda _event: self._on_key(-1))
        self.bind("<Right>", lambda _event: self._on_key(1))
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _clamp_index(self, index: int) -> int:
        return min(len(self.labels) - 1, max(0, int(index)))

    def _tab_bounds(self, index: int) -> tuple[float, float]:
        if not 0 <= index < len(self.labels):
            raise IndexError(index)
        width = max(self.winfo_width(), self.winfo_reqwidth())
        tab_width = float(width) / len(self.labels)
        return index * tab_width, (index + 1) * tab_width

    def select(
        self,
        index: int,
        *,
        animate: bool = True,
        notify: bool = True,
    ) -> None:
        target = self._clamp_index(index)
        if target == self.selected_index:
            return
        self.cancel_animation()
        self.selected_index = target
        self._redraw()
        target_x = self._target_pill_x(target)
        if animate and self.animation_ms > 0 and self._pill_initialized:
            self._animate_pill_to(target_x)
        else:
            self._pill_x = target_x
            self._redraw()
        if notify and self.command is not None:
            self.command(target)

    def set_palette(self, palette: Mapping[str, str]) -> None:
        self._palette = {**LIGHT_NAV_PALETTE, **palette}
        self.configure(background=self._palette["background"])
        self._redraw()

    def cancel_animation(self) -> None:
        callback_id = self._animation_after_id
        self._animation_after_id = None
        if callback_id is not None:
            try:
                self.after_cancel(callback_id)
            except tk.TclError:
                pass

    def _on_key(self, delta: int) -> str:
        self.select(self.selected_index + delta)
        return "break"

    def _on_click(self, event) -> str:
        self.focus_set()
        for index in range(len(self.labels)):
            left, right = self._tab_bounds(index)
            if left <= float(event.x) < right or (
                index == len(self.labels) - 1 and float(event.x) == right
            ):
                self.select(index)
                break
        return "break"

    def _target_pill_x(self, index: int) -> float:
        left, right = self._tab_bounds(self._clamp_index(index))
        return (left + right) / 2.0

    def _animate_pill_to(self, target_x: float) -> None:
        start_x = self._pill_x
        if abs(target_x - start_x) < 0.01:
            self._pill_x = target_x
            self._redraw()
            return
        steps = 10
        interval = max(1, int(round(self.animation_ms / steps)))

        def advance(step: int = 1) -> None:
            self._animation_after_id = None
            if self._destroyed or not self.winfo_exists():
                return
            fraction = min(1.0, step / steps)
            self._pill_x = start_x + (target_x - start_x) * fraction
            self._redraw()
            if step >= steps:
                self._pill_x = target_x
                return
            self._animation_after_id = self.after(
                interval,
                lambda: advance(step + 1),
            )

        self._animation_after_id = self.after(interval, advance)

    def _redraw(self) -> None:
        if self._destroyed or not self.winfo_exists():
            return
        self.delete("all")
        width = max(self.winfo_width(), self.winfo_reqwidth())
        height = max(self.winfo_height(), self.winfo_reqheight())
        inset = 1.0
        capsule_left, capsule_top = inset, inset
        capsule_right, capsule_bottom = width - inset, height - inset
        self._rounded_box(
            capsule_left,
            capsule_top,
            capsule_right,
            capsule_bottom,
            fill=self._palette["capsule"],
            outline=self._palette["capsule_outline"],
            tags="capsule",
        )
        if not self._pill_initialized:
            self._pill_x = self._target_pill_x(self.selected_index)
            self._pill_initialized = True
        tab_width = float(width) / len(self.labels)
        pill_half_width = max(4.0, tab_width / 2.0 - 3.0)
        pill_left = max(capsule_left + 2.0, self._pill_x - pill_half_width)
        pill_right = min(capsule_right - 2.0, self._pill_x + pill_half_width)
        pill_top = capsule_top + 3.0
        pill_bottom = capsule_bottom - 3.0
        self._rounded_box(
            pill_left,
            pill_top,
            pill_right,
            pill_bottom,
            fill=self._palette["pill"],
            outline=self._palette["pill"],
            tags="pill",
        )
        highlight_y = pill_top + 2.0
        self.create_line(
            pill_left + min(8.0, pill_half_width),
            highlight_y,
            pill_right - min(8.0, pill_half_width),
            highlight_y,
            fill=self._palette["pill_highlight"],
            width=1,
            tags=("pill", "pill-highlight"),
        )
        for index, label in enumerate(self.labels):
            left, right = self._tab_bounds(index)
            self.create_text(
                (left + right) / 2,
                height / 2,
                text=label,
                fill=(
                    self._palette["active_text"]
                    if index == self.selected_index
                    else self._palette["text"]
                ),
                font=("Tahoma", 9, "bold" if index == self.selected_index else "normal"),
                tags=("tab", f"tab-{index}", "label"),
            )
        if self._focused:
            self.create_rectangle(
                inset + 2,
                inset + 2,
                width - inset - 2,
                height - inset - 2,
                outline=self._palette["focus"],
                width=2,
                tags="focus",
            )

    def _rounded_box(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        *,
        fill: str,
        outline: str,
        tags: str | tuple[str, ...],
    ) -> None:
        radius = max(1.0, min((bottom - top) / 2.0, (right - left) / 2.0))
        self.create_rectangle(
            left + radius,
            top,
            right - radius,
            bottom,
            fill=fill,
            outline=outline,
            tags=tags,
        )
        if bottom - top > radius * 2.0:
            self.create_rectangle(
                left,
                top + radius,
                right,
                bottom - radius,
                fill=fill,
                outline=outline,
                tags=tags,
            )
        self.create_oval(
            left,
            top,
            left + radius * 2.0,
            bottom,
            fill=fill,
            outline=outline,
            tags=tags,
        )
        self.create_oval(
            right - radius * 2.0,
            top,
            right,
            bottom,
            fill=fill,
            outline=outline,
            tags=tags,
        )

    def _on_configure(self, _event=None) -> None:
        self._redraw()

    def _on_focus_in(self, _event=None) -> None:
        self._focused = True
        self._redraw()

    def _on_focus_out(self, _event=None) -> None:
        self._focused = False
        self._redraw()

    def _on_destroy(self, _event=None) -> None:
        self.cancel_animation()
        self._destroyed = True


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
        palette: Mapping[str, str] | None = None,
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
            background=(palette or LIGHT_SLIDER_PALETTE)["background"],
        )
        self._palette = dict(LIGHT_SLIDER_PALETTE)
        if palette:
            self._palette.update(palette)
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

    def set_palette(self, palette: Mapping[str, str]) -> None:
        self._palette = {**LIGHT_SLIDER_PALETTE, **palette}
        self.configure(background=self._palette["background"])
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
            fill=self._palette["rail"],
            outline=self._palette["rail_outline"],
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
                outline=self._palette["focus"],
                width=2,
                tags="focus",
            )
        if self._hovered and not self._pressed:
            self.create_oval(
                thumb_x - 10,
                center_y - 10,
                thumb_x + 10,
                center_y + 10,
                fill=self._palette["hover"],
                outline="",
                tags="halo",
            )

        shadow_y = 2 if not self._pressed else 1
        self.create_oval(
            thumb_x - 8,
            center_y - 6 + shadow_y,
            thumb_x + 8,
            center_y + 10,
            fill=self._palette["shadow"],
            outline="",
            tags=("thumb", "thumb_shadow"),
        )
        body_fill = self._palette["thumb_pressed"] if self._pressed else self._palette["thumb"]
        self.create_oval(
            thumb_x - 8,
            center_y - 8,
            thumb_x + 8,
            center_y + 8,
            fill=body_fill,
            outline=self._palette["thumb_outline"],
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
                outline=self._palette["highlight"],
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
                fill=self._palette["bubble"],
                outline=self._palette["thumb_pressed"],
                tags="bubble",
            )
            self.create_text(
                thumb_x,
                bubble_y,
                text=text,
                fill=self._palette["text"],
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
