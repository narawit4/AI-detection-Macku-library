import threading
import unittest

from ai_targeting import Detection, DetectionFrameSnapshot
from overlay import (
    DetectionOverlay,
    OverlayBox,
    OverlaySetupError,
    WDA_EXCLUDEFROMCAPTURE,
    WS_EX_LAYERED,
    WS_EX_NOACTIVATE,
    WS_EX_TOOLWINDOW,
    WS_EX_TRANSPARENT,
    Win32OverlayAdapter,
    project_overlay_boxes,
)


class OverlayProjectionTests(unittest.TestCase):
    def test_projects_all_boxes_and_emphasizes_selected_index(self):
        frame = DetectionFrameSnapshot(
            3,
            10.0,
            (
                Detection(1, 2, 30, 40, 0.8, 0),
                Detection(100, 110, 130, 150, 0.9, 7),
            ),
            1,
        )

        self.assertEqual(
            project_overlay_boxes(frame, now=10.1),
            (
                OverlayBox(1, 2, 30, 40, 2),
                OverlayBox(100, 110, 130, 150, 4),
            ),
        )

    def test_absent_or_stale_frame_projects_no_boxes(self):
        frame = DetectionFrameSnapshot(1, 10.0, (), None)

        self.assertEqual(project_overlay_boxes(None, 10.0), ())
        self.assertEqual(project_overlay_boxes(frame, 10.151), ())

    def test_exact_freshness_boundary_and_future_timestamp_are_accepted(self):
        frame = DetectionFrameSnapshot(
            1,
            0.0,
            (Detection(5, 6, 25, 26, 0.7, 0),),
            None,
        )
        expected = (OverlayBox(5, 6, 25, 26, 2),)

        self.assertEqual(project_overlay_boxes(frame, 0.150), expected)
        self.assertEqual(project_overlay_boxes(frame, -1.0), expected)

    def test_head_boxes_can_be_hidden_without_hiding_player_boxes(self):
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (
                Detection(1, 2, 30, 40, 0.8, 0),
                Detection(100, 110, 130, 150, 0.9, 7),
            ),
            1,
        )

        boxes = project_overlay_boxes(frame, 10.0, show_heads=False)

        self.assertEqual(boxes, (OverlayBox(1, 2, 30, 40, 2),))


class FakeNativeApi:
    def __init__(self, *, style=0x1000, root_hwnd=1234):
        self.style = style
        self.root_hwnd = root_hwnd
        self.error = 0
        self.calls = []
        self.get_result = style
        self.set_result = style
        self.affinity_result = True
        self.get_error = 0
        self.set_error = 0
        self.affinity_error = 0

    def GetAncestor(self, hwnd, flags):
        self.calls.append(("get-root", hwnd, flags))
        return self.root_hwnd

    def GetWindowLongPtrW(self, hwnd, index):
        self.calls.append(("get-style", hwnd, index))
        self.error = self.get_error
        return self.get_result

    def SetWindowLongPtrW(self, hwnd, index, value):
        self.calls.append(("set-style", hwnd, index, value))
        self.error = self.set_error
        return self.set_result

    def SetWindowDisplayAffinity(self, hwnd, affinity):
        self.calls.append(("set-affinity", hwnd, affinity))
        self.error = self.affinity_error
        return self.affinity_result

    def set_last_error(self, value):
        self.error = value

    def get_last_error(self):
        return self.error


class Win32OverlayAdapterTests(unittest.TestCase):
    def make_adapter(self, api):
        return Win32OverlayAdapter(
            user32=api,
            get_last_error=api.get_last_error,
            set_last_error=api.set_last_error,
        )

    def test_configure_adds_all_required_styles_before_capture_exclusion(self):
        api = FakeNativeApi(style=0x1000)

        self.make_adapter(api).configure(1234)

        expected_styles = (
            0x1000
            | WS_EX_TRANSPARENT
            | WS_EX_TOOLWINDOW
            | WS_EX_LAYERED
            | WS_EX_NOACTIVATE
        )
        self.assertEqual(
            api.calls,
            [
                ("get-root", 1234, 2),
                ("get-style", 1234, -20),
                ("set-style", 1234, -20, expected_styles),
                ("set-affinity", 1234, WDA_EXCLUDEFROMCAPTURE),
            ],
        )

    def test_configure_resolves_tk_child_to_top_level_before_native_setup(self):
        api = FakeNativeApi(style=0x1000, root_hwnd=5678)

        self.make_adapter(api).configure(1234)

        expected_styles = (
            0x1000
            | WS_EX_TRANSPARENT
            | WS_EX_TOOLWINDOW
            | WS_EX_LAYERED
            | WS_EX_NOACTIVATE
        )
        self.assertEqual(
            api.calls,
            [
                ("get-root", 1234, 2),
                ("get-style", 5678, -20),
                ("set-style", 5678, -20, expected_styles),
                ("set-affinity", 5678, WDA_EXCLUDEFROMCAPTURE),
            ],
        )

    def test_zero_get_style_with_last_error_fails_before_later_operations(self):
        api = FakeNativeApi(style=0)
        api.get_result = 0
        api.get_error = 5

        with self.assertRaisesRegex(
            OverlaySetupError, "GetWindowLongPtrW failed.*5"
        ):
            self.make_adapter(api).configure(1234)

        self.assertEqual(
            api.calls,
            [("get-root", 1234, 2), ("get-style", 1234, -20)],
        )

    def test_zero_set_style_with_last_error_fails_before_capture_exclusion(self):
        api = FakeNativeApi()
        api.set_result = 0
        api.set_error = 87

        with self.assertRaisesRegex(
            OverlaySetupError, "SetWindowLongPtrW failed.*87"
        ):
            self.make_adapter(api).configure(1234)

        self.assertEqual(
            [call[0] for call in api.calls],
            ["get-root", "get-style", "set-style"],
        )

    def test_false_capture_exclusion_result_fails_closed(self):
        api = FakeNativeApi()
        api.affinity_result = False
        api.affinity_error = 5

        with self.assertRaisesRegex(
            OverlaySetupError, "SetWindowDisplayAffinity failed.*5"
        ):
            self.make_adapter(api).configure(1234)

        self.assertEqual(api.calls[-1], ("set-affinity", 1234, 0x11))


class FakeWindow:
    def __init__(self, calls, failures=None):
        self.calls = calls
        self.failures = failures or {}
        self.destroyed = False
        self.visible = False

    def _fail(self, operation):
        error = self.failures.get(operation)
        if error is not None:
            raise error

    def withdraw(self):
        self.calls.append("withdraw")
        self.visible = False
        self._fail("withdraw")

    def overrideredirect(self, value):
        self.calls.append(("borderless", value))

    def attributes(self, name, value):
        self.calls.append((name, value))
        self._fail(name)

    def winfo_screenwidth(self):
        return 1920

    def winfo_screenheight(self):
        return 1080

    def geometry(self, value):
        self.calls.append(("geometry", value))

    def update_idletasks(self):
        self.calls.append("update-idletasks")

    def winfo_id(self):
        return 1234

    def deiconify(self):
        self.calls.append("deiconify")
        self.visible = True
        self._fail("deiconify")

    def lift(self):
        self.calls.append("lift")
        self._fail("lift")

    def destroy(self):
        self.destroyed = True
        self.visible = False
        self.calls.append("destroy")
        self._fail("destroy")


class FakeCanvas:
    def __init__(self, window, calls, options, failures=None):
        self.window = window
        self.calls = calls
        self.options = options
        self.failures = failures or {}
        self.items = []

    def pack(self, **options):
        self.pack_options = options

    def delete(self, tag):
        self.calls.append(("delete", tag))
        error = self.failures.get("delete")
        if error is not None:
            raise error
        self.items = []

    def create_rectangle(self, *coords, **options):
        self.items.append((coords, options))


class RecordingAdapter:
    def __init__(self, calls, error=None):
        self.calls = calls
        self.error = error
        self.handles = []

    def configure(self, hwnd):
        self.calls.append("configure-win32")
        self.handles.append(hwnd)
        if self.error:
            raise self.error


class DetectionOverlayTests(unittest.TestCase):
    def make_overlay(
        self,
        *,
        setup_error=None,
        window_failures=None,
        canvas_failures=None,
    ):
        calls = []
        window = FakeWindow(calls, window_failures)
        canvases = []

        def canvas_factory(owner, **options):
            canvas = FakeCanvas(owner, calls, options, canvas_failures)
            canvases.append(canvas)
            return canvas

        adapter = RecordingAdapter(calls, setup_error)
        overlay = DetectionOverlay(
            object(),
            window_factory=lambda _root: window,
            canvas_factory=canvas_factory,
            win32_adapter=adapter,
        )
        return overlay, window, canvases, adapter, calls

    def test_show_configures_hidden_centered_window_before_deiconifying(self):
        overlay, window, canvases, adapter, calls = self.make_overlay()

        overlay.show()

        self.assertEqual(adapter.handles, [window.winfo_id()])
        self.assertLess(calls.index("withdraw"), calls.index("configure-win32"))
        self.assertLess(calls.index("configure-win32"), calls.index("deiconify"))
        self.assertEqual(
            calls[:5],
            [
                "withdraw",
                ("borderless", True),
                ("-topmost", True),
                ("-transparentcolor", "#010203"),
                ("geometry", "320x320+800+380"),
            ],
        )
        self.assertEqual(
            canvases[0].options,
            {
                "width": 320,
                "height": 320,
                "background": "#010203",
                "highlightthickness": 0,
            },
        )
        self.assertEqual(canvases[0].pack_options, {"fill": "both", "expand": True})
        self.assertTrue(overlay.visible)

    def test_setup_failure_destroys_window_and_remains_hidden(self):
        error = OverlaySetupError("capture exclusion failed")
        overlay, window, _canvases, adapter, calls = self.make_overlay(
            setup_error=error
        )

        with self.assertRaisesRegex(OverlaySetupError, "capture exclusion failed"):
            overlay.show()

        self.assertEqual(adapter.handles, [1234])
        self.assertTrue(window.destroyed)
        self.assertFalse(overlay.visible)
        self.assertNotIn("deiconify", calls)

    def test_tk_setup_failure_destroys_window_and_remains_hidden(self):
        overlay, window, _canvases, adapter, calls = self.make_overlay(
            window_failures={"-transparentcolor": RuntimeError("color failed")}
        )

        with self.assertRaisesRegex(RuntimeError, "color failed"):
            overlay.show()

        self.assertTrue(window.destroyed)
        self.assertFalse(overlay.visible)
        self.assertEqual(adapter.handles, [])
        self.assertNotIn("deiconify", calls)

    def test_deiconify_failure_destroys_window_and_detaches_ownership(self):
        overlay, window, _canvases, _adapter, calls = self.make_overlay(
            window_failures={"deiconify": RuntimeError("deiconify failed")}
        )

        with self.assertRaisesRegex(RuntimeError, "deiconify failed"):
            overlay.show()

        self.assertTrue(window.destroyed)
        self.assertFalse(window.visible)
        self.assertFalse(overlay.visible)
        overlay.close()
        overlay.close()
        self.assertEqual(calls.count("destroy"), 1)

    def test_lift_failure_destroys_visible_window_and_detaches_ownership(self):
        overlay, window, _canvases, _adapter, calls = self.make_overlay(
            window_failures={"lift": RuntimeError("lift failed")}
        )

        with self.assertRaisesRegex(RuntimeError, "lift failed"):
            overlay.show()

        self.assertTrue(window.destroyed)
        self.assertFalse(window.visible)
        self.assertFalse(overlay.visible)
        self.assertLess(calls.index("deiconify"), calls.index("lift"))
        overlay.close()
        overlay.close()
        self.assertEqual(calls.count("destroy"), 1)

    def test_render_replaces_boxes_with_red_selected_width(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        first = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(1, 2, 30, 40, 0.8, 0),),
            None,
        )
        second = DetectionFrameSnapshot(
            2,
            10.1,
            (
                Detection(5, 6, 25, 26, 0.7, 0),
                Detection(100, 110, 130, 150, 0.9, 7),
            ),
            1,
        )

        overlay.render(first, now=10.0)
        overlay.render(second, now=10.2)

        self.assertEqual(
            canvases[0].items,
            [
                (
                    (5, 6, 25, 26),
                    {"outline": "#ff2b2b", "width": 2, "tags": ("detection",)},
                ),
                (
                    (100, 110, 130, 150),
                    {"outline": "#ff2b2b", "width": 4, "tags": ("detection",)},
                ),
            ],
        )

    def test_render_uses_requested_box_color(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(1, 2, 30, 40, 0.8, 0),),
            None,
        )

        overlay.render(frame, now=10.0, color="#00cc88")

        self.assertEqual(canvases[0].items[0][1]["outline"], "#00cc88")

    def test_stale_render_clears_existing_boxes_while_window_stays_visible(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(1, 2, 30, 40, 0.8, 0),),
            None,
        )
        overlay.render(frame, now=10.0)

        overlay.render(frame, now=10.151)

        self.assertEqual(canvases[0].items, [])
        self.assertTrue(overlay.visible)

    def test_hide_clears_before_withdrawing_and_can_be_shown_again(self):
        overlay, _window, _canvases, adapter, calls = self.make_overlay()
        overlay.show()
        calls.clear()

        overlay.hide()
        overlay.show()

        self.assertEqual(calls[:2], [("delete", "detection"), "withdraw"])
        self.assertEqual(adapter.handles, [1234])
        self.assertTrue(overlay.visible)

    def test_close_clears_then_destroys_once_and_prevents_reopening(self):
        overlay, window, _canvases, _adapter, calls = self.make_overlay()
        overlay.show()
        calls.clear()

        overlay.close()
        overlay.close()

        self.assertEqual(calls, [("delete", "detection"), "destroy"])
        self.assertTrue(window.destroyed)
        self.assertFalse(overlay.visible)
        with self.assertRaisesRegex(OverlaySetupError, "Overlay is closed"):
            overlay.show()

    def test_close_logs_cleanup_failures_and_still_prevents_reopening(self):
        overlay, window, _canvases, _adapter, calls = self.make_overlay(
            window_failures={"destroy": RuntimeError("destroy failed")},
            canvas_failures={"delete": RuntimeError("delete failed")},
        )
        overlay.show()
        calls.clear()

        with self.assertLogs("overlay", level="ERROR") as logged:
            overlay.close()

        self.assertEqual(calls, [("delete", "detection"), "destroy"])
        self.assertTrue(window.destroyed)
        self.assertFalse(overlay.visible)
        self.assertEqual(len(logged.output), 2)
        with self.assertRaisesRegex(OverlaySetupError, "Overlay is closed"):
            overlay.show()

    def test_worker_thread_is_rejected_before_any_tk_access(self):
        overlay, _window, _canvases, adapter, calls = self.make_overlay()
        errors = []

        worker = threading.Thread(
            target=lambda: self._capture_error(overlay.show, errors)
        )
        worker.start()
        worker.join()

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("main thread", str(errors[0]).lower())
        self.assertEqual(calls, [])
        self.assertEqual(adapter.handles, [])

    @staticmethod
    def _capture_error(operation, errors):
        try:
            operation()
        except Exception as error:
            errors.append(error)


if __name__ == "__main__":
    unittest.main()
