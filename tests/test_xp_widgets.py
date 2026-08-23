import tkinter as tk
import unittest

from xp_widgets import XPGlossySlider


class _SliderTestCase(unittest.TestCase):
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
        slider = XPGlossySlider(self.root, **options)
        slider.pack()
        self.root.update_idletasks()
        return slider


class XPGlossySliderValueTests(_SliderTestCase):
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
