import tkinter as tk
import unittest
from types import SimpleNamespace

from liquid_widgets import LiquidNavigation, LiquidSlider


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
        slider.event_generate("<Button-1>", x=14, y=17)
        self.root.update()
        pressed_fill = slider.itemcget("thumb_body", "fill")
        self.assertEqual(pressed_fill, "#356FAF")
        slider.focus_set()
        slider.event_generate("<FocusIn>")
        self.root.update()
        self.assertTrue(slider.find_withtag("focus"))

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

    def test_pointer_selects_the_hit_tab(self):
        left, right = self.nav._tab_bounds(1)
        self.nav._on_click(SimpleNamespace(x=(left + right) / 2))
        self.assertEqual(self.nav.selected_index, 1)

    def test_palette_redraws_capsule_and_active_pill(self):
        palette = {
            "background": "#171B22", "capsule": "#285A91",
            "capsule_outline": "#8CBCEB", "pill": "#4C8CCC",
            "pill_highlight": "#FFFFFF", "text": "#E7ECF3",
            "active_text": "#FFFFFF", "focus": "#F2B84B",
        }
        self.nav.set_palette(palette)
        self.root.update_idletasks()
        self.assertEqual(self.nav.cget("background"), "#171B22")
        self.assertEqual(self.nav.itemcget("capsule", "fill"), "#285A91")
        self.assertEqual(self.nav.itemcget("pill", "fill"), "#4C8CCC")

    def test_new_selection_replaces_obsolete_animation(self):
        self.nav.select(2)
        first = self.nav._animation_after_id
        self.nav.select(1)
        self.assertIsNotNone(self.nav._animation_after_id)
        self.assertNotEqual(self.nav._animation_after_id, first)
        self.nav.cancel_animation()
        self.assertIsNone(self.nav._animation_after_id)

    def test_scheduler_failure_snaps_to_target_and_still_notifies(self):
        self.nav._redraw()

        def failing_after(*_args):
            raise tk.TclError("scheduler unavailable")

        self.nav.after = failing_after
        try:
            self.nav.select(2)
        except tk.TclError as exc:
            self.fail(f"selection leaked animation failure: {exc}")

        self.assertEqual(self.nav.selected_index, 2)
        self.assertEqual(self.selected, [2])
        self.assertIsNone(self.nav._animation_after_id)
        self.assertAlmostEqual(
            self.nav._pill_x,
            self.nav._target_pill_x(2),
        )

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
        for item_id in self.nav.find_withtag("capsule") + self.nav.find_withtag("pill"):
            if self.nav.type(item_id) == "rectangle":
                coordinates = self.nav.coords(item_id)
                self.assertLess(coordinates[1], coordinates[3])

    def test_destroy_cancels_animation(self):
        self.nav.select(2)
        self.nav.destroy()
        self.root.update()
        self.assertIsNone(self.nav._animation_after_id)
