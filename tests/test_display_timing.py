import unittest

from jitter_app.device.display_timing import RuntimeCadence, cadence_from_refresh, detect_runtime_cadence


class DisplayTimingTests(unittest.TestCase):
    def test_valid_refresh_drives_capture_and_double_rate_servo(self):
        self.assertEqual(
            cadence_from_refresh(144),
            RuntimeCadence(display_hz=144, capture_fps=144, servo_hz=288),
        )

    def test_caps_and_fallback_are_exact(self):
        self.assertEqual(cadence_from_refresh(360), RuntimeCadence(360, 240, 480))
        for raw in (None, True, float("nan"), 23.99, 500.01, "bad"):
            with self.subTest(raw=raw):
                self.assertEqual(cadence_from_refresh(raw), RuntimeCadence(None, 120, 240))

    def test_valid_boundaries_and_rounding(self):
        self.assertEqual(cadence_from_refresh(24), RuntimeCadence(24, 24, 120))
        self.assertEqual(cadence_from_refresh(500), RuntimeCadence(500, 240, 480))
        self.assertEqual(cadence_from_refresh(143.6), RuntimeCadence(144, 144, 288))

    def test_win32_success_writes_current_frequency(self):
        class FakeUser32:
            def EnumDisplaySettingsW(self, device, mode_num, mode_ptr):
                mode_ptr._obj.dmDisplayFrequency = 165
                return 1

        self.assertEqual(
            detect_runtime_cadence(FakeUser32()), RuntimeCadence(165, 165, 330)
        )

    def test_win32_false_returns_fallback(self):
        class FakeUser32:
            def EnumDisplaySettingsW(self, device, mode_num, mode_ptr):
                return 0

        self.assertEqual(detect_runtime_cadence(FakeUser32()), RuntimeCadence(None, 120, 240))

    def test_win32_error_returns_fallback(self):
        class FakeUser32:
            def EnumDisplaySettingsW(self, device, mode_num, mode_ptr):
                raise OSError("not available")

        self.assertEqual(detect_runtime_cadence(FakeUser32()), RuntimeCadence(None, 120, 240))


if __name__ == "__main__":
    unittest.main()
