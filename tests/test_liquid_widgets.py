import tkinter as tk
import unittest
from types import SimpleNamespace

import liquid_widgets
from liquid_widgets import LiquidIconButton, LiquidNavigation, LiquidSlider


DARK_ICON_PALETTE = {
    "background": "#111827", "surface": "#243247",
    "surface_hover": "#31506B", "surface_pressed": "#1B2A3A",
    "surface_disabled": "#364152", "border": "#4B6380",
    "icon": "#EAF7FF", "icon_disabled": "#93A4B8",
    "highlight": "#719AB8", "focus": "#FFE08A",
}


def canvas_text_font_family(canvas, tag):
    for item_id in canvas.find_withtag(tag):
        if canvas.type(item_id) == "text":
            return canvas.tk.splitlist(canvas.itemcget(item_id, "font"))[0]
    raise AssertionError(f"no text item found for Canvas tag {tag!r}")


class _SliderTestCase(unittest.TestCase):
    slider_type = LiquidSlider

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self):
        self.root.destroy()

    def make_slider(self, **kwargs):
        options = {
            "from_": 0,
            "to": 100,
            "resolution": 1,
            "width": 220,
            "height": 34,
        }
        options.update(kwargs)
        slider = self.slider_type(self.root, **options)
        slider.pack()
        self.root.update_idletasks()
        return slider


class LiquidSliderValueTests(_SliderTestCase):
    def test_rejects_invalid_range_and_resolution(self):
        with self.assertRaises(ValueError):
            self.make_slider(from_=1, to=1)
        with self.assertRaises(ValueError):
            self.make_slider(resolution=0)

    def test_set_clamps_and_snaps_without_calling_command(self):
        emitted = []
        slider = self.make_slider(
            from_=0.1,
            to=60,
            resolution=0.1,
            command=emitted.append,
        )
        slider.set(22.26)
        self.assertAlmostEqual(slider.get(), 22.3)
        slider.set(-5)
        self.assertAlmostEqual(slider.get(), 0.1)
        slider.set(100)
        self.assertAlmostEqual(slider.get(), 60.0)
        self.assertEqual(emitted, [])

    def test_formats_integral_and_fractional_resolutions_cleanly(self):
        integer = self.make_slider(resolution=1)
        decimal = self.make_slider(from_=0.1, to=60, resolution=0.1)
        self.assertEqual(integer._format_value(22), "22")
        self.assertEqual(decimal._format_value(22.3), "22.3")

    def test_value_position_conversion_covers_range_endpoints_and_midpoint(self):
        slider = self.make_slider()
        left, right = slider._rail_bounds()
        self.assertAlmostEqual(slider._value_to_x(0), left)
        self.assertAlmostEqual(slider._value_to_x(50), (left + right) / 2)
        self.assertAlmostEqual(slider._value_to_x(100), right)
        self.assertEqual(slider._x_to_value(left - 50), 0)
        self.assertEqual(slider._x_to_value(right + 50), 100)


class LiquidSliderInteractionTests(_SliderTestCase):
    def test_pointer_click_and_drag_emit_snapped_values(self):
        emitted = []
        slider = self.make_slider(command=emitted.append)
        self.root.deiconify()
        self.root.update()
        left, right = slider._rail_bounds()
        slider.event_generate(
            "<Button-1>",
            x=int((left + right) / 2),
            y=17,
        )
        slider.event_generate("<B1-Motion>", x=int(right), y=17)
        slider.event_generate("<ButtonRelease-1>", x=int(right), y=17)
        self.root.update()
        self.assertEqual(slider.get(), 100)
        self.assertEqual(emitted[-1], "100")

    def test_arrow_home_and_end_keys_update_and_emit(self):
        emitted = []
        slider = self.make_slider(command=emitted.append)
        self.root.deiconify()
        self.root.update()
        slider.focus_force()
        slider.set(50)
        slider.event_generate("<Right>")
        slider.event_generate("<Up>")
        slider.event_generate("<Home>")
        slider.event_generate("<End>")
        self.root.update()
        self.assertEqual(slider.get(), 100)
        self.assertEqual(emitted, ["51", "52", "0", "100"])

    def test_hover_press_focus_and_bubble_change_canvas_state(self):
        slider = self.make_slider()
        self.root.deiconify()
        self.root.update()
        slider.event_generate("<Enter>", x=14, y=17)
        self.root.update()
        self.assertTrue(slider.find_withtag("halo"))
        self.assertTrue(slider.find_withtag("bubble"))
        self.assertEqual(canvas_text_font_family(slider, "bubble"), "Segoe UI")
        slider.event_generate("<Button-1>", x=14, y=17)
        self.root.update()
        pressed_fill = slider.itemcget("thumb_body", "fill")
        self.assertEqual(pressed_fill, "#55DDF6")
        slider.focus_set()
        slider.event_generate("<FocusIn>")
        self.root.update()
        self.assertTrue(slider.find_withtag("focus-ring"))

    def test_slider_palette_updates_rail_fill_thumb_and_disabled_colors(self):
        """Fails if the slider stops rendering liquid palette roles."""
        slider = self.make_slider()
        slider.set_palette({
            "background": "#111827", "rail": "#27364A", "fill": "#63E6FF",
            "thumb": "#E8FBFF", "thumb_border": "#63E6FF",
            "halo": "#315F70", "text": "#EDF7FF", "bubble": "#24384A",
            "bubble_text": "#EDF7FF", "focus": "#FFE08A",
            "disabled": "#536174", "disabled_text": "#8C99AA",
        })
        self.assertEqual(slider.itemcget("rail", "fill"), "#27364A")
        self.assertEqual(slider.itemcget("fill", "fill"), "#63E6FF")
        self.assertEqual(slider.itemcget("thumb", "fill"), "#E8FBFF")
        slider.configure(state=tk.DISABLED)
        slider._redraw()
        self.assertEqual(slider.itemcget("rail", "fill"), "#536174")
        self.assertEqual(slider.itemcget("thumb", "fill"), "#536174")

    def test_destroy_cancels_pending_bubble_hide(self):
        slider = self.make_slider()
        self.root.deiconify()
        self.root.update()
        slider.event_generate("<Enter>", x=14, y=17)
        slider.event_generate("<Leave>", x=14, y=17)
        self.root.update_idletasks()
        self.assertIsNotNone(slider._bubble_after_id)
        slider.destroy()
        self.root.update()
        self.assertTrue(slider._destroyed)
        self.assertIsNone(slider._bubble_after_id)

    def test_scheduler_teardown_does_not_leak_from_bubble_hide(self):
        slider = self.make_slider()

        def failing_after(*_args):
            raise tk.TclError("scheduler unavailable")

        slider.after = failing_after
        try:
            slider._schedule_hide_bubble()
        except tk.TclError as exc:
            self.fail(f"bubble scheduling leaked teardown failure: {exc}")
        self.assertIsNone(slider._bubble_after_id)

    def test_grab_failure_does_not_leave_slider_pressed(self):
        slider = self.make_slider()

        def failing_grab():
            raise tk.TclError("grab unavailable")

        slider.grab_set = failing_grab
        try:
            slider._on_press(SimpleNamespace(x=17, y=17))
        except tk.TclError as exc:
            self.fail(f"slider leaked grab failure: {exc}")
        self.assertFalse(slider._pressed)


class LiquidNavigationTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.selected = []
        self.nav = LiquidNavigation(
            self.root,
            labels=("Control", "Motion", "Advanced"),
            command=self.selected.append,
            width=330,
        )
        self.nav.pack()
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def make_vertical_nav(self):
        return LiquidNavigation(
            self.root,
            labels=("Control", "Motion", "Advanced"),
            command=self.selected.append,
            orientation="vertical",
            width=152,
            height=216,
        )

    def test_rejects_unknown_orientation(self):
        with self.assertRaisesRegex(ValueError, "orientation"):
            LiquidNavigation(
                self.root,
                labels=("A",),
                command=lambda _index: None,
                orientation="diagonal",
            )

    def test_vertical_navigation_stacks_destinations_and_moves_lens_on_y_axis(self):
        nav = self.make_vertical_nav()
        nav.pack()
        self.root.update()
        first = nav._target_lens_position(0)
        last = nav._target_lens_position(2)
        self.assertEqual(first[0], last[0])
        self.assertLess(first[1], last[1])

    def test_vertical_pointer_selects_the_destination_hit_on_y_axis(self):
        nav = self.make_vertical_nav()
        nav.pack()
        self.root.update()
        left, top, right, bottom = nav._item_bounds(2)
        nav._on_click(SimpleNamespace(x=(left + right) / 2, y=(top + bottom) / 2))
        self.assertEqual(nav.selected_index, 2)

    def test_vertical_up_down_home_end_and_activation(self):
        nav = self.make_vertical_nav()
        nav.pack()
        self.root.deiconify()
        self.root.update()
        nav.focus_force()
        nav.select(1, animate=False)
        nav.event_generate("<Down>")
        nav.event_generate("<Up>")
        nav.event_generate("<End>")
        nav.event_generate("<Return>")
        self.root.update()
        self.assertEqual(nav.selected_index, 2)
        self.assertEqual(self.selected[-1], 2)

    def test_select_clamps_index_and_notifies_once(self):
        self.nav.select(2, animate=False)
        self.assertEqual(self.nav.selected_index, 2)
        self.assertEqual(self.selected, [2])
        self.nav.select(99, animate=False)
        self.assertEqual(self.nav.selected_index, 2)
        self.assertEqual(self.selected, [2])

    def test_arrow_keys_select_adjacent_tabs(self):
        self.nav._on_key(1)
        self.nav._on_key(1)
        self.nav._on_key(1)
        self.assertEqual(self.nav.selected_index, 2)
        self.nav._on_key(-1)
        self.assertEqual(self.nav.selected_index, 1)

    def test_home_and_end_keys_select_boundary_tabs(self):
        self.root.deiconify()
        self.root.update()
        self.nav.focus_force()
        self.nav.event_generate("<End>")
        self.root.update()
        self.assertEqual(self.nav.selected_index, 2)
        self.nav.event_generate("<Home>")
        self.root.update()
        self.assertEqual(self.nav.selected_index, 0)

    def test_enter_and_space_activate_selected_tab_once_per_key(self):
        """Fails if keyboard activation is missing or double-notifies."""
        self.root.deiconify()
        self.root.update()
        self.nav.select(1, animate=False)
        self.selected.clear()
        self.nav.focus_force()
        self.root.update()

        self.nav.event_generate("<Return>")
        self.root.update()
        self.nav.event_generate("<space>")
        self.root.update()

        self.assertEqual(self.nav.selected_index, 1)
        self.assertEqual(self.selected, [1, 1])

    def test_pointer_selects_the_hit_tab(self):
        left, right = self.nav._tab_bounds(1)
        self.nav._on_click(SimpleNamespace(x=(left + right) / 2))
        self.assertEqual(self.nav.selected_index, 1)

    def test_navigation_palette_updates_glass_lens_and_focus_ring(self):
        """Fails if navigation drops a liquid surface, lens, or focus role."""
        palette = {
            "background": "#111827", "surface": "#1B2638",
            "surface_highlight": "#33445E", "border": "#45566F",
            "lens": "#63E6FF", "lens_highlight": "#B8F6FF",
            "text": "#EDF7FF", "selected_text": "#08212A",
            "focus": "#FFE08A",
        }
        self.nav.set_palette(palette)
        self.nav._on_focus_in()
        self.root.update_idletasks()
        self.assertEqual(self.nav.cget("background"), "#111827")
        self.assertEqual(self.nav.itemcget("glass", "fill"), "#1B2638")
        self.assertEqual(self.nav.itemcget("lens", "fill"), "#63E6FF")
        self.assertEqual(self.nav.itemcget("focus-ring", "outline"), "#FFE08A")
        self.assertEqual(
            canvas_text_font_family(self.nav, "label"),
            "Segoe UI",
        )

    def test_navigation_focus_ring_follows_rounded_control_shape(self):
        self.nav._on_focus_in()
        focus_items = self.nav.find_withtag("focus-ring")
        self.assertTrue(focus_items)
        self.assertIn("oval", {self.nav.type(item) for item in focus_items})

    def test_new_selection_replaces_obsolete_animation(self):
        self.nav.select(2)
        first = self.nav._animation_after_id
        self.nav.select(1)
        self.assertIsNotNone(self.nav._animation_after_id)
        self.assertNotEqual(self.nav._animation_after_id, first)
        self.nav.cancel_animation()
        self.assertIsNone(self.nav._animation_after_id)

    def test_scheduler_failure_snaps_but_drawing_tcl_error_propagates(self):
        self.nav._redraw()

        def failing_after(*_args):
            raise tk.TclError("scheduler unavailable")

        original_after = self.nav.after
        self.nav.after = failing_after
        try:
            try:
                self.nav.select(2)
            except tk.TclError as exc:
                self.fail(f"selection leaked animation failure: {exc}")
        finally:
            self.nav.after = original_after

        self.assertEqual(self.nav.selected_index, 2)
        self.assertEqual(self.selected, [2])
        self.assertIsNone(self.nav._animation_after_id)
        self.assertAlmostEqual(
            self.nav._pill_x,
            self.nav._target_pill_x(2),
        )

        self.nav.selected_index = 0

        def failing_redraw():
            raise tk.TclError("drawing failed")

        original_redraw = self.nav._redraw
        self.nav._redraw = failing_redraw
        try:
            with self.assertRaisesRegex(tk.TclError, "drawing failed"):
                self.nav.select(1)
        finally:
            self.nav._redraw = original_redraw

    def test_unexpected_animation_error_is_not_masked_as_teardown(self):
        self.nav._redraw()

        def failing_after(*_args):
            raise RuntimeError("unexpected scheduler contract violation")

        self.nav.after = failing_after
        with self.assertRaisesRegex(RuntimeError, "contract violation"):
            self.nav.select(2)

    def test_rapid_selection_finishes_at_latest_target(self):
        self.nav._redraw()
        cancelled = []
        original_after_cancel = self.nav.after_cancel

        def recording_after_cancel(callback_id):
            cancelled.append(callback_id)
            return original_after_cancel(callback_id)

        self.nav.after_cancel = recording_after_cancel
        self.nav.select(2)
        obsolete_callback_id = self.nav._animation_after_id
        self.nav.select(1)
        self.assertIn(obsolete_callback_id, cancelled)
        self.root.after(self.nav.animation_ms * 3, self.root.quit)
        self.root.mainloop()
        self.assertIsNone(self.nav._animation_after_id)
        self.assertAlmostEqual(
            self.nav._pill_x,
            self.nav._target_pill_x(self.nav.selected_index),
        )

    def test_rounded_shapes_do_not_contain_collapsed_center_rectangles(self):
        self.nav._redraw()
        for item_id in self.nav.find_withtag("glass") + self.nav.find_withtag("lens"):
            if self.nav.type(item_id) == "rectangle":
                coordinates = self.nav.coords(item_id)
                self.assertLess(coordinates[1], coordinates[3])

    def test_destroy_cancels_animation(self):
        self.nav.select(2)
        self.nav.destroy()
        self.root.update()
        self.assertIsNone(self.nav._animation_after_id)

    def test_exported_widget_classes_and_palettes_have_no_legacy_names(self):
        """Fails if a legacy public liquid-widget symbol is reintroduced."""
        legacy_prefix = "X" + "P"
        for name, value in vars(liquid_widgets).items():
            if isinstance(value, type) or name.endswith("PALETTE"):
                self.assertNotIn(legacy_prefix, name)


class LiquidIconButtonTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.calls = []
        self.button = LiquidIconButton(
            self.root,
            icon="↻",
            accessible_name="Reconnect Makcu",
            command=lambda: self.calls.append("called"),
        )
        self.button.pack()
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def test_click_enter_and_space_activate_when_enabled(self):
        """Fails if an enabled action stops invoking its real command."""
        self.root.deiconify()
        self.root.update()
        self.button.focus_force()
        self.button._activate()
        self.button.event_generate("<Return>")
        self.button.event_generate("<space>")
        self.root.update()
        self.assertEqual(self.calls, ["called", "called", "called"])

    def test_disabled_button_does_not_activate(self):
        """Fails if disabled actions can still invoke their command."""
        self.button.set_enabled(False)
        self.button._activate()
        self.assertEqual(self.calls, [])

    def test_disable_and_reenable_preserves_actual_keyboard_focus(self):
        """Fails if a temporary disabled state loses the retained focus ring."""
        self.root.deiconify()
        self.root.update()
        self.button.focus_force()
        self.root.update()
        self.assertIs(self.root.focus_get(), self.button)
        self.assertTrue(self.button.find_withtag("focus-ring"))

        self.button.set_enabled(False)
        self.root.update()
        self.assertIs(self.root.focus_get(), self.button)
        self.assertFalse(self.button.find_withtag("focus-ring"))

        self.button.set_enabled(True)
        self.root.update()
        self.assertIs(self.root.focus_get(), self.button)
        self.assertTrue(self.button.find_withtag("focus-ring"))

        other = tk.Entry(self.root)
        other.pack()
        other.focus_force()
        self.root.update()
        self.assertIs(self.root.focus_get(), other)
        self.assertFalse(self.button.find_withtag("focus-ring"))

    def test_grab_failure_does_not_leave_button_pressed(self):
        def failing_grab():
            raise tk.TclError("grab unavailable")

        self.button.grab_set = failing_grab
        self.button._on_press(SimpleNamespace(x=17, y=17))
        self.assertFalse(self.button._pressed)

    def test_palette_redraws_surface_icon_and_focus(self):
        """Fails if a palette update does not redraw visible icon controls."""
        self.button.set_palette(DARK_ICON_PALETTE)
        self.button._on_focus_in()
        self.assertEqual(self.button.itemcget("surface", "fill"), "#243247")
        self.assertEqual(self.button.itemcget("icon", "fill"), "#EAF7FF")
        self.assertEqual(
            self.button.itemcget("focus-ring", "outline"),
            "#FFE08A",
        )
        self.assertEqual(
            canvas_text_font_family(self.button, "icon"),
            "Segoe UI",
        )
        self.assertEqual(self.button.accessible_name, "Reconnect Makcu")

    def test_pointer_release_activates_and_clears_pressed_state(self):
        """Fails if a normal release leaves a button visibly pressed."""
        self.button._on_enter()
        self.button._on_press(SimpleNamespace(x=17, y=17))
        self.button._on_release(SimpleNamespace(x=17, y=17))
        self.assertEqual(self.calls, ["called"])
        self.assertFalse(self.button._pressed)

    def test_command_destroying_button_leaves_no_pressed_state_or_callback(self):
        """Fails if command teardown is followed by stale press cleanup."""
        self.button.command = self.button.destroy
        self.button._on_press(SimpleNamespace(x=17, y=17))
        self.button._on_release(SimpleNamespace(x=17, y=17))
        self.root.update()
        self.assertTrue(self.button._destroyed)
        self.assertFalse(self.button._pressed)

    def test_throwing_command_clears_pressed_state_before_propagating(self):
        """Fails if a command error leaves the action visibly pressed."""
        def raise_command_error():
            raise RuntimeError("command failure")

        self.button.command = raise_command_error
        self.button._on_press(SimpleNamespace(x=17, y=17))
        with self.assertRaisesRegex(RuntimeError, "command failure"):
            self.button._on_release(SimpleNamespace(x=17, y=17))
        self.assertFalse(self.button._pressed)

    def test_release_outside_does_not_activate_and_releases_press_state(self):
        """Fails if an outside release activates or leaves a stuck button."""
        self.button._on_press(SimpleNamespace(x=17, y=17))
        self.button._on_leave()
        self.button._on_release(SimpleNamespace(x=34, y=17))
        self.assertEqual(self.calls, [])
        self.assertFalse(self.button._pressed)
