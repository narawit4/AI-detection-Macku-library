from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from functools import partial
import math
import tkinter as tk
from typing import Callable, Mapping


DEFAULT_NAV_PALETTE = {
    "background": "#F2F7FA", "surface": "#E5F0F5",
    "surface_highlight": "#FFFFFF", "border": "#B9CBD5",
    "lens": "#55DDF6", "lens_highlight": "#C7F8FF",
    "text": "#263640", "selected_text": "#07252C", "focus": "#8B5CF6",
}


DEFAULT_SLIDER_PALETTE = {
    "background": "#F2F7FA", "rail": "#C9D9E1", "fill": "#55DDF6",
    "thumb": "#F8FEFF", "thumb_border": "#33BDD8", "halo": "#B7EFF8",
    "text": "#263640", "bubble": "#244653", "bubble_text": "#FFFFFF",
    "focus": "#8B5CF6", "disabled": "#A9B6BC", "disabled_text": "#7A878D",
}


DEFAULT_ICON_PALETTE = {
    "background": "#F2F7FA", "surface": "#E5F0F5",
    "surface_hover": "#D6F5FA", "surface_pressed": "#B7EFF8",
    "surface_disabled": "#D5E0E5", "border": "#B9CBD5",
    "icon": "#263640", "icon_disabled": "#7A878D",
    "highlight": "#FFFFFF", "focus": "#8B5CF6",
}


class LiquidNavigation(tk.Canvas):
    """Keyboard-accessible animated navigation for the liquid dashboard."""

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
        orientation: str = "horizontal",
    ) -> None:
        self.labels = tuple(labels)
        if not self.labels or any(
            not isinstance(label, str) or not label.strip()
            for label in self.labels
        ):
            raise ValueError("navigation labels must be non-empty")
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")
        self.orientation = orientation

        self._palette = dict(DEFAULT_NAV_PALETTE)
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
        self._pill_y = 0.0
        self._pill_initialized = False

        self.bind("<Configure>", self._on_configure)
        self.bind("<Button-1>", self._on_click)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        if self.orientation == "horizontal":
            self.bind("<Left>", lambda _event: self._on_key(-1))
            self.bind("<Right>", lambda _event: self._on_key(1))
        else:
            self.bind("<Up>", lambda _event: self._on_key(-1))
            self.bind("<Down>", lambda _event: self._on_key(1))
        self.bind("<Home>", lambda _event: self._on_boundary(0))
        self.bind("<End>", lambda _event: self._on_boundary(len(self.labels) - 1))
        self.bind("<Return>", self._on_activate)
        self.bind("<space>", self._on_activate)
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _clamp_index(self, index: int) -> int:
        return min(len(self.labels) - 1, max(0, int(index)))

    def _item_bounds(self, index: int) -> tuple[float, float, float, float]:
        if not 0 <= index < len(self.labels):
            raise IndexError(index)
        width = max(self.winfo_width(), self.winfo_reqwidth())
        height = max(self.winfo_height(), self.winfo_reqheight())
        if self.orientation == "horizontal":
            item_width = float(width) / len(self.labels)
            return index * item_width, 0.0, (index + 1) * item_width, float(height)
        item_height = float(height) / len(self.labels)
        return 0.0, index * item_height, float(width), (index + 1) * item_height

    def _tab_bounds(self, index: int) -> tuple[float, float]:
        """Retain the horizontal bounds helper for existing consumers."""
        left, _top, right, _bottom = self._item_bounds(index)
        return left, right

    def _target_lens_position(self, index: int) -> tuple[float, float]:
        left, top, right, bottom = self._item_bounds(self._clamp_index(index))
        return (left + right) / 2.0, (top + bottom) / 2.0

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
        try:
            self._redraw()
            target_x, target_y = self._target_lens_position(target)
            if animate and self.animation_ms > 0 and self._pill_initialized:
                self._animate_pill_to(target_x, target_y)
            else:
                self._pill_x = target_x
                self._pill_y = target_y
                self._redraw()
        finally:
            if notify and self.command is not None:
                self.command(target)

    def set_palette(self, palette: Mapping[str, str]) -> None:
        self._palette = {**DEFAULT_NAV_PALETTE, **palette}
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

    def _on_boundary(self, index: int) -> str:
        self.select(index)
        return "break"

    def _on_activate(self, _event=None) -> str:
        if self.command is not None:
            self.command(self.selected_index)
        return "break"

    def _on_click(self, event) -> str:
        self.focus_set()
        for index in range(len(self.labels)):
            left, top, right, bottom = self._item_bounds(index)
            x = float(getattr(event, "x", 0.0))
            y = float(getattr(event, "y", 0.0))
            if (
                left <= x < right
                and top <= y < bottom
            ) or (
                index == len(self.labels) - 1
                and (
                    (self.orientation == "horizontal" and x == right)
                    or (self.orientation == "vertical" and y == bottom)
                )
            ):
                self.select(index)
                break
        return "break"

    def _target_pill_x(self, index: int) -> float:
        return self._target_lens_position(index)[0]

    def _animate_pill_to(self, target_x: float, target_y: float) -> None:
        start_x = self._pill_x
        start_y = self._pill_y
        if abs(target_x - start_x) < 0.01 and abs(target_y - start_y) < 0.01:
            self._pill_x = target_x
            self._pill_y = target_y
            self._redraw()
            return
        steps = 10
        interval = max(1, int(round(self.animation_ms / steps)))
        self._schedule_pill_step(
            start_x, start_y, target_x, target_y, steps, interval, 1,
        )

    def _schedule_pill_step(
        self,
        start_x: float,
        start_y: float,
        target_x: float,
        target_y: float,
        steps: int,
        interval: int,
        step: int,
    ) -> None:
        callback = partial(
            self._advance_pill,
            start_x,
            start_y,
            target_x,
            target_y,
            steps,
            interval,
            step,
        )
        try:
            self._animation_after_id = self.after(interval, callback)
        except tk.TclError:
            self._animation_after_id = None
            self._pill_x = target_x
            self._pill_y = target_y
            self._redraw()

    def _advance_pill(
        self,
        start_x: float,
        start_y: float,
        target_x: float,
        target_y: float,
        steps: int,
        interval: int,
        step: int,
    ) -> None:
        self._animation_after_id = None
        if self._destroyed or not self.winfo_exists():
            return
        fraction = min(1.0, step / steps)
        self._pill_x = start_x + (target_x - start_x) * fraction
        self._pill_y = start_y + (target_y - start_y) * fraction
        self._redraw()
        if step >= steps:
            self._pill_x = target_x
            self._pill_y = target_y
            return
        self._schedule_pill_step(
            start_x,
            start_y,
            target_x,
            target_y,
            steps,
            interval,
            step + 1,
        )

    def _redraw(self) -> None:
        if self._destroyed or not self.winfo_exists():
            return
        self.delete("all")
        width = max(self.winfo_width(), self.winfo_reqwidth())
        height = max(self.winfo_height(), self.winfo_reqheight())
        inset = 1.0
        glass_left, glass_top = inset, inset
        glass_right, glass_bottom = width - inset, height - inset
        self._rounded_box(
            glass_left,
            glass_top,
            glass_right,
            glass_bottom,
            fill=self._palette["surface"],
            outline="",
            tags="glass",
        )
        if not self._pill_initialized:
            self._pill_x, self._pill_y = self._target_lens_position(
                self.selected_index,
            )
            self._pill_initialized = True
        if self.orientation == "horizontal":
            item_width = float(width) / len(self.labels)
            lens_half_width = max(4.0, item_width / 2.0 - 4.0)
            pill_left = max(glass_left + 3.0, self._pill_x - lens_half_width)
            pill_right = min(glass_right - 3.0, self._pill_x + lens_half_width)
            pill_top = glass_top + 3.0
            pill_bottom = glass_bottom - 3.0
        else:
            item_height = float(height) / len(self.labels)
            lens_half_height = max(4.0, item_height / 2.0 - 4.0)
            pill_left = glass_left + 3.0
            pill_right = glass_right - 3.0
            pill_top = max(glass_top + 3.0, self._pill_y - lens_half_height)
            pill_bottom = min(glass_bottom - 3.0, self._pill_y + lens_half_height)
        self._rounded_box(
            pill_left,
            pill_top,
            pill_right,
            pill_bottom,
            fill=self._palette["lens"],
            outline="",
            tags="lens",
        )
        for index, label in enumerate(self.labels):
            left, top, right, bottom = self._item_bounds(index)
            self.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text=label,
                fill=(
                    self._palette["selected_text"]
                    if index == self.selected_index
                    else self._palette["text"]
                ),
                font=("Segoe UI", 9, "bold" if index == self.selected_index else "normal"),
                tags=("tab", f"tab-{index}", "label"),
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
        self.create_polygon(
            left + radius, top,
            right - radius, top,
            right, top,
            right, top + radius,
            right, bottom - radius,
            right, bottom,
            right - radius, bottom,
            left + radius, bottom,
            left, bottom,
            left, bottom - radius,
            left, top + radius,
            left, top,
            left + radius, top,
            fill=fill,
            outline=outline,
            smooth=True,
            splinesteps=24,
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


class LiquidSlider(tk.Canvas):
    """Exact-value Canvas slider for the liquid dashboard."""
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
            background=(palette or DEFAULT_SLIDER_PALETTE)["background"],
        )
        self._palette = dict(DEFAULT_SLIDER_PALETTE)
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
        self._palette = {**DEFAULT_SLIDER_PALETTE, **palette}
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

    def _is_disabled(self) -> bool:
        return str(self.cget("state")) == tk.DISABLED

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

    def _redraw(self) -> None:
        if self._destroyed or not self.winfo_exists():
            return
        self.delete("all")
        height = max(self.winfo_height(), self.winfo_reqheight())
        left, right = self._rail_bounds()
        center_y = min(height - 10, 21)
        thumb_x = self._value_to_x(self._value)
        disabled = self._is_disabled()
        rail_color = self._palette["disabled"] if disabled else self._palette["rail"]
        fill_color = self._palette["disabled"] if disabled else self._palette["fill"]
        thumb_color = self._palette["disabled"] if disabled else (
            self._palette["fill"] if self._pressed else self._palette["thumb"]
        )
        thumb_border = self._palette["disabled"] if disabled else self._palette["thumb_border"]

        self._rounded_box(
            left,
            center_y - 4,
            right,
            center_y + 4,
            fill=rail_color,
            outline=rail_color,
            tags="rail",
        )
        fill_right = max(left, thumb_x)
        if fill_right - left >= 4:
            self._rounded_box(
                left,
                center_y - 3,
                fill_right,
                center_y + 3,
                fill=fill_color,
                outline=fill_color,
                tags="fill",
            )
        else:
            self.create_rectangle(
                left,
                center_y - 3,
                fill_right,
                center_y + 3,
                fill=fill_color,
                outline=fill_color,
                tags="fill",
            )

        if self._focused and not disabled:
            self.create_oval(
                thumb_x - 11,
                center_y - 11,
                thumb_x + 11,
                center_y + 11,
                outline=self._palette["focus"],
                width=2,
                tags="focus-ring",
            )
        if self._hovered and not self._pressed and not disabled:
            self.create_oval(
                thumb_x - 10,
                center_y - 10,
                thumb_x + 10,
                center_y + 10,
                fill=self._palette["halo"],
                outline="",
                tags="halo",
            )

        self.create_oval(
            thumb_x - 8,
            center_y - 8,
            thumb_x + 8,
            center_y + 8,
            fill=thumb_color,
            outline=thumb_border,
            width=1,
            tags=("thumb", "thumb_body"),
        )
        if not self._pressed and not disabled:
            self.create_arc(
                thumb_x - 6,
                center_y - 6,
                thumb_x + 6,
                center_y + 5,
                start=20,
                extent=140,
                style="arc",
                outline=self._palette["halo"],
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
                outline=fill_color,
                tags="bubble",
            )
            self.create_text(
                thumb_x,
                bubble_y,
                text=text,
                fill=(self._palette["disabled_text"] if disabled else self._palette["bubble_text"]),
                font=("Segoe UI", 8),
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
        try:
            self.grab_set()
        except tk.TclError:
            self._pressed = False
            self._redraw()
            return "break"
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

    def cancel_pending_callbacks(self) -> None:
        self._cancel_bubble_hide()

    def _schedule_hide_bubble(self) -> None:
        self._cancel_bubble_hide()
        try:
            self._bubble_after_id = self.after(450, self._hide_bubble)
        except tk.TclError:
            self._bubble_after_id = None

    def _hide_bubble(self) -> None:
        self._bubble_after_id = None
        if self._destroyed:
            return
        self._bubble_visible = False
        self._redraw()

    def _on_configure(self, _event=None) -> None:
        self._redraw()

    def _on_destroy(self, _event=None) -> None:
        self.cancel_pending_callbacks()
        self._destroyed = True


class LiquidIconButton(tk.Canvas):
    """Compact, keyboard-accessible Canvas button for liquid icon actions."""

    def __init__(
        self,
        parent,
        *,
        icon: str,
        accessible_name: str,
        command: Callable[[], None],
        palette: Mapping[str, str] | None = None,
        size: int = 34,
    ) -> None:
        self._palette = dict(DEFAULT_ICON_PALETTE)
        if palette:
            self._palette.update(palette)
        self.size = int(size)
        if self.size <= 0:
            raise ValueError("icon button size must be positive")
        super().__init__(
            parent,
            width=self.size,
            height=self.size,
            highlightthickness=0,
            borderwidth=0,
            takefocus=True,
            background=self._palette["background"],
        )
        self.icon = icon
        self.accessible_name = accessible_name
        self.command = command
        self._enabled = True
        self._destroyed = False
        self._hovered = False
        self._pressed = False
        self._focused = False
        self._has_pointer_grab = False

        self.bind("<Configure>", self._on_configure)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Return>", self._on_key_activate)
        self.bind("<space>", self._on_key_activate)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Destroy>", self._on_destroy, add="+")

    def set_palette(self, palette: Mapping[str, str]) -> None:
        self._palette = {**DEFAULT_ICON_PALETTE, **palette}
        self.configure(background=self._palette["background"])
        self._redraw()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not self._enabled:
            self._release_pointer_grab()
            self._pressed = False
            self._hovered = False
        self.configure(state=tk.NORMAL if self._enabled else tk.DISABLED)
        self._redraw()

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

    def _redraw(self) -> None:
        if self._destroyed or not self.winfo_exists():
            return
        self.delete("all")
        width = max(self.winfo_width(), self.winfo_reqwidth())
        height = max(self.winfo_height(), self.winfo_reqheight())
        inset = 1.0
        if not self._enabled:
            surface = self._palette["surface_disabled"]
            icon_color = self._palette["icon_disabled"]
        elif self._pressed:
            surface = self._palette["surface_pressed"]
            icon_color = self._palette["icon"]
        elif self._hovered:
            surface = self._palette["surface_hover"]
            icon_color = self._palette["icon"]
        else:
            surface = self._palette["surface"]
            icon_color = self._palette["icon"]
        self._rounded_box(
            inset,
            inset,
            width - inset,
            height - inset,
            fill=surface,
            outline=self._palette["border"],
            tags="surface",
        )
        self.create_arc(
            inset + 3,
            inset + 3,
            width - inset - 3,
            height - inset - 5,
            start=20,
            extent=140,
            style="arc",
            outline=self._palette["highlight"],
            width=1,
            tags="surface-highlight",
        )
        self.create_text(
            width / 2,
            height / 2,
            text=self.icon,
            fill=icon_color,
            font=("Segoe UI", max(10, round(self.size * 0.45)), "bold"),
            tags="icon",
        )
        if self._focused and self._enabled:
            self._rounded_box(
                2,
                2,
                width - 2,
                height - 2,
                fill="",
                outline=self._palette["focus"],
                tags="focus-ring",
            )

    def _activate(self) -> None:
        if self._enabled and not self._destroyed:
            self.command()

    def _on_key_activate(self, _event=None) -> str:
        self._activate()
        return "break"

    def _on_enter(self, _event=None) -> None:
        if self._enabled:
            self._hovered = True
            self._redraw()

    def _on_leave(self, _event=None) -> None:
        if self._enabled:
            self._hovered = False
            self._redraw()

    def _on_press(self, _event=None) -> str:
        if self._enabled:
            self.focus_set()
            self._pressed = True
            try:
                self.grab_set()
                self._has_pointer_grab = True
            except (RuntimeError, tk.TclError):
                self._has_pointer_grab = False
                self._pressed = False
            self._redraw()
        return "break"

    def _on_release(self, event=None) -> str:
        was_pressed = self._pressed
        self._release_pointer_grab()
        self._pressed = False
        if not self._destroyed:
            self._redraw()
        if (
            was_pressed
            and self._enabled
            and not self._destroyed
            and self._release_is_inside(event)
        ):
            self._activate()
        return "break"

    def _on_focus_in(self, _event=None) -> None:
        if self._enabled:
            self._focused = True
            self._redraw()

    def _on_focus_out(self, _event=None) -> None:
        self._focused = False
        self._redraw()

    def _release_is_inside(self, event) -> bool:
        if event is None:
            return False
        width = max(self.winfo_width(), self.winfo_reqwidth())
        height = max(self.winfo_height(), self.winfo_reqheight())
        return (
            0 <= float(getattr(event, "x", -1)) < width
            and 0 <= float(getattr(event, "y", -1)) < height
        )

    def _release_pointer_grab(self) -> None:
        if not self._has_pointer_grab:
            return
        self._has_pointer_grab = False
        try:
            self.grab_release()
        except (RuntimeError, tk.TclError):
            pass

    def _on_configure(self, _event=None) -> None:
        self._redraw()

    def _on_destroy(self, _event=None) -> None:
        self._destroyed = True
        self._pressed = False
        self._release_pointer_grab()
