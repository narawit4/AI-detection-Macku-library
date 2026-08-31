import threading
import unittest

import jitter_app.presentation.overlay as overlay_module
from jitter_app.ai.targeting import Detection, DetectionFrameSnapshot
from jitter_app.presentation.overlay import (
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

    def test_hidden_earlier_head_does_not_reindex_selected_player(self):
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (
                Detection(100, 110, 130, 150, 0.9, 7),
                Detection(1, 2, 30, 40, 0.8, 0),
            ),
            1,
        )

        self.assertEqual(
            project_overlay_boxes(frame, 10.0, show_heads=False),
            (OverlayBox(1, 2, 30, 40, 4),),
        )

    def test_projection_scales_snapshot_geometry_to_changed_canvas(self):
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(480, 270, 1440, 810, 0.8, 0),),
            0,
            1920,
            1080,
        )

        boxes = project_overlay_boxes(
            frame,
            10.0,
            canvas_width=2560,
            canvas_height=1440,
        )

        self.assertEqual(
            (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2),
            (640.0, 360.0, 1920.0, 1080.0),
        )

    def test_projection_clamps_boxes_and_labels_to_canvas(self):
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(-20, -10, 1940, 1090, 0.8, 7),),
            0,
            1920,
            1080,
        )

        boxes = project_overlay_boxes(
            frame,
            10.0,
            canvas_width=1920,
            canvas_height=1080,
            label_mode="class",
        )

        self.assertEqual(
            (
                boxes[0].x1,
                boxes[0].y1,
                boxes[0].x2,
                boxes[0].y2,
                boxes[0].label,
            ),
            (0.0, 0.0, 1920.0, 1080.0, "HEAD"),
        )

    def test_invalid_canvas_and_empty_projection_preserve_original_selection(
        self,
    ):
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (
                Detection(-20, 10, -1, 40, 0.8, 0),
                Detection(100, 100, 200, 200, 0.9, 7),
            ),
            1,
            1920,
            1080,
        )
        for width, height in (
            (0, 1080),
            (-1, 1080),
            (True, 1080),
            (1920, 0),
            (1920, False),
        ):
            with self.subTest(canvas=(width, height)):
                self.assertEqual(
                    project_overlay_boxes(
                        frame,
                        10.0,
                        canvas_width=width,
                        canvas_height=height,
                    ),
                    (),
                )

        boxes = project_overlay_boxes(
            frame,
            10.0,
            canvas_width=1920,
            canvas_height=1080,
        )
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].width, 4)

        selected_was_omitted = DetectionFrameSnapshot(
            1,
            10.0,
            frame.detections,
            0,
            1920,
            1080,
        )
        boxes = project_overlay_boxes(
            selected_was_omitted,
            10.0,
            canvas_width=1920,
            canvas_height=1080,
        )
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].width, 2)


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
    def __init__(self, calls, failures=None, screen_size=(1920, 1080)):
        self.calls = calls
        self.failures = failures or {}
        self.screen_size = screen_size
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
        return self.screen_size[0]

    def winfo_screenheight(self):
        return self.screen_size[1]

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
        self.bbox_value = None
        self.moves = []

    def pack(self, **options):
        self.pack_options = options

    def delete(self, tag):
        self.calls.append(("delete", tag))
        error = self.failures.get("delete")
        if error is not None:
            raise error
        self.items = [
            item
            for item in self.items
            if tag not in item[1].get("tags", ())
        ]

    def configure(self, **options):
        self.calls.append(("canvas-configure", options))
        error = self.failures.get("configure")
        if error is not None:
            raise error
        self.options.update(options)

    def create_rectangle(self, *coords, **options):
        self.items.append((coords, options))

    def create_text(self, *coords, **options):
        self.items.append((coords, options))
        return len(self.items) - 1

    def bbox(self, _item_id):
        return self.bbox_value

    def move(self, item_id, dx, dy):
        self.moves.append((item_id, dx, dy))
        coords, options = self.items[item_id]
        self.items[item_id] = (
            (coords[0] + dx, coords[1] + dy),
            options,
        )


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
        screen_size=(1920, 1080),
    ):
        calls = []
        window = FakeWindow(calls, window_failures, screen_size)
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

    def test_show_configures_hidden_full_screen_window_before_deiconifying(self):
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
                ("geometry", "1920x1080+0+0"),
            ],
        )
        self.assertEqual(
            canvases[0].options,
            {
                "width": 1920,
                "height": 1080,
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
            1920,
            1080,
        )
        second = DetectionFrameSnapshot(
            2,
            10.1,
            (
                Detection(5, 6, 25, 26, 0.7, 0),
                Detection(100, 110, 130, 150, 0.9, 7),
            ),
            1,
            1920,
            1080,
        )

        overlay.render(first, now=10.0)
        overlay.render(second, now=10.2)

        self.assertEqual(
            canvases[0].items,
            [
                (
                    (5.0, 6.0, 25.0, 26.0),
                    {"outline": "#ff2b2b", "width": 2, "tags": ("detection",)},
                ),
                (
                    (100.0, 110.0, 130.0, 150.0),
                    {"outline": "#ff2b2b", "width": 4, "tags": ("detection",)},
                ),
            ],
        )

    def test_overlay_style_can_hide_players_without_hiding_heads(self):
        self.assertTrue(
            hasattr(overlay_module, "OverlayStyle"),
            "OverlayStyle must expose immutable per-frame customization",
        )
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (
                Detection(1, 2, 30, 40, 0.8, 0),
                Detection(100, 110, 130, 150, 0.9, 7),
            ),
            1,
            1920,
            1080,
        )
        style = overlay_module.OverlayStyle(show_players=False)

        overlay.render(frame, now=10.0, style=style)

        self.assertEqual(len(canvases[0].items), 1)
        self.assertEqual(
            canvases[0].items[0][0],
            (100.0, 110.0, 130.0, 150.0),
        )

    def test_overlay_style_controls_normal_and_selected_box_width(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (
                Detection(1, 2, 30, 40, 0.8, 0),
                Detection(100, 110, 130, 150, 0.9, 7),
            ),
            1,
            1920,
            1080,
        )
        try:
            style = overlay_module.OverlayStyle(box_width=6)
        except TypeError as exc:
            self.fail(f"OverlayStyle must accept box_width: {exc}")

        overlay.render(frame, now=10.0, style=style)

        self.assertEqual(
            [item[1]["width"] for item in canvases[0].items],
            [6, 8],
        )

    def test_selected_box_remains_emphasized_at_maximum_base_width(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (
                Detection(1, 2, 30, 40, 0.8, 0),
                Detection(100, 110, 130, 150, 0.9, 7),
            ),
            1,
            1920,
            1080,
        )

        overlay.render(
            frame,
            now=10.0,
            style=overlay_module.OverlayStyle(box_width=8),
        )

        self.assertEqual(
            [item[1]["width"] for item in canvases[0].items],
            [8, 10],
        )

    def test_class_confidence_mode_labels_each_detection(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (
                Detection(1, 2, 30, 40, 0.806, 0),
                Detection(100, 110, 130, 150, 0.934, 7),
            ),
            None,
            1920,
            1080,
        )
        try:
            style = overlay_module.OverlayStyle(
                label_mode="class_confidence"
            )
        except TypeError as exc:
            self.fail(f"OverlayStyle must accept label_mode: {exc}")

        overlay.render(frame, now=10.0, style=style)

        labels = [
            item[1]["text"]
            for item in canvases[0].items
            if "text" in item[1]
        ]
        self.assertEqual(labels, ["PLAYER 81%", "HEAD 93%"])
        self.assertTrue(all(
            item[1]["tags"] == ("detection",)
            for item in canvases[0].items
        ))

    def test_class_mode_omits_confidence_from_detection_label(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(1, 2, 30, 40, 0.806, 0),),
            None,
            1920,
            1080,
        )

        overlay.render(
            frame,
            now=10.0,
            style=overlay_module.OverlayStyle(label_mode="class"),
        )

        labels = [
            item[1]["text"]
            for item in canvases[0].items
            if "text" in item[1]
        ]
        self.assertEqual(labels, ["PLAYER"])

    def test_detection_labels_move_minimally_to_stay_on_screen(self):
        cases = (
            ((-10, 10, 30, 30), (10, 0)),
            ((80, 10, 110, 30), (-10, 0)),
            ((10, -5, 30, 15), (0, 5)),
            ((10, 85, 30, 105), (0, -5)),
        )
        for bounds, expected_move in cases:
            with self.subTest(bounds=bounds):
                overlay, _window, canvases, _adapter, _calls = self.make_overlay(
                    screen_size=(100, 100)
                )
                overlay.show()
                canvases[0].bbox_value = bounds
                frame = DetectionFrameSnapshot(
                    1,
                    10.0,
                    (Detection(10, 10, 30, 30, 0.8, 0),),
                    None,
                    100,
                    100,
                )

                overlay.render(
                    frame,
                    now=10.0,
                    style=overlay_module.OverlayStyle(
                        label_mode="class",
                        hud_visible=False,
                    ),
                )

                self.assertEqual(
                    canvases[0].moves,
                    [(1, expected_move[0], expected_move[1])],
                )

    def test_detection_label_never_maps_an_unsupported_class_to_player(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(1, 2, 30, 40, 0.806, 99),),
            None,
            1920,
            1080,
        )

        overlay.render(
            frame,
            now=10.0,
            style=overlay_module.OverlayStyle(label_mode="class"),
        )

        labels = [
            item[1]["text"]
            for item in canvases[0].items
            if "text" in item[1]
        ]
        self.assertEqual(labels, [])

    def test_render_projects_full_source_frame_over_entire_canvas(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay(
            screen_size=(2560, 1440)
        )
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(0, 0, 2560, 1440, 0.8, 0),),
            0,
            2560,
            1440,
        )

        overlay.render(frame, now=10.0)

        self.assertEqual(
            canvases[0].items[0][0],
            (0.0, 0.0, 2560.0, 1440.0),
        )

    def test_render_passes_changed_canvas_geometry_into_projection(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay(
            screen_size=(2560, 1440)
        )
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(480, 270, 1440, 810, 0.8, 0),),
            0,
            1920,
            1080,
        )

        overlay.render(frame, now=10.0)

        self.assertEqual(
            canvases[0].items[0][0],
            (640.0, 360.0, 1920.0, 1080.0),
        )

    def test_show_refreshes_geometry_and_projection_after_display_change(self):
        overlay, window, canvases, _adapter, calls = self.make_overlay()
        overlay.show()
        overlay.hide()
        window.screen_size = (2560, 1440)
        calls.clear()

        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(480, 270, 1440, 810, 0.8, 0),),
            0,
            1920,
            1080,
        )
        overlay.render(frame, now=10.0)

        self.assertIn(("geometry", "2560x1440+0+0"), calls)
        self.assertEqual(canvases[0].options["width"], 2560)
        self.assertEqual(canvases[0].options["height"], 1440)
        self.assertEqual(
            canvases[0].items[0][0],
            (640.0, 360.0, 1920.0, 1080.0),
        )

    def test_render_uses_requested_box_color(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(1, 2, 30, 40, 0.8, 0),),
            None,
            1920,
            1080,
        )

        overlay.render(frame, now=10.0, color="#00cc88")

        self.assertEqual(canvases[0].items[0][1]["outline"], "#00cc88")

    def test_render_draws_runtime_hud_and_selected_head_lock(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(100, 110, 130, 150, 0.9, 7),),
            0,
            1920,
            1080,
        )

        try:
            overlay.render(
                frame,
                now=10.0,
                color="#00cc88",
                runtime=("120 FPS", "DirectML", "1.5×"),
            )
        except TypeError as exc:
            self.fail(f"render must accept runtime status: {exc}")

        self.assertEqual(len(canvases[0].items), 2)
        self.assertEqual(
            canvases[0].items[0][1]["tags"],
            ("detection",),
        )
        self.assertEqual(
            canvases[0].items[-1],
            (
                (8, 8),
                {
                    "anchor": "nw",
                    "fill": "#00cc88",
                    "font": ("Consolas", 10, "bold"),
                    "text": (
                        "AI RUNTIME\n"
                        "FPS: 120 FPS\n"
                        "PROVIDER: DirectML\n"
                        "ZOOM: 1.5×\n"
                        "LOCK: HEAD"
                    ),
                    "tags": ("runtime",),
                },
            ),
        )

    def test_runtime_hud_supports_all_screen_corners_with_exact_offsets(self):
        expected = {
            "top_left": ((12, 34), "nw"),
            "top_right": ((1908, 34), "ne"),
            "bottom_left": ((12, 1046), "sw"),
            "bottom_right": ((1908, 1046), "se"),
        }

        for corner, (coords, anchor) in expected.items():
            with self.subTest(corner=corner):
                overlay, _window, canvases, _adapter, _calls = self.make_overlay()
                overlay.show()
                try:
                    style = overlay_module.OverlayStyle(
                        hud_corner=corner,
                        hud_offset_x=12,
                        hud_offset_y=34,
                    )
                except TypeError as exc:
                    self.fail(f"OverlayStyle must accept HUD placement: {exc}")

                overlay.render(
                    None,
                    now=10.0,
                    runtime=("120 FPS", "DirectML", "1.0Ã—"),
                    style=style,
                )

                self.assertEqual(canvases[0].items[-1][0], coords)
                self.assertEqual(canvases[0].items[-1][1]["anchor"], anchor)

    def test_runtime_hud_offsets_are_clamped_to_the_current_screen(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        style = overlay_module.OverlayStyle(
            hud_corner="bottom_right",
            hud_offset_x=5000,
            hud_offset_y=-5,
        )

        overlay.render(
            None,
            now=10.0,
            runtime=("120 FPS", "DirectML", "1.0Ã—"),
            style=style,
        )

        self.assertEqual(canvases[0].items[-1][0], (0, 1080))

    def test_runtime_hud_moves_minimally_to_keep_full_text_bbox_on_screen(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        canvases[0].bbox_value = (-200, 900, 100, 1120)
        style = overlay_module.OverlayStyle(
            hud_corner="bottom_right",
            hud_offset_x=5000,
            hud_offset_y=-5,
        )

        overlay.render(
            None,
            now=10.0,
            runtime=("120 FPS", "DirectML", "1.0Ã—"),
            style=style,
        )

        self.assertEqual(canvases[0].moves, [(0, 200, -40)])
        left, top, right, bottom = (
            value + delta
            for value, delta in zip(
                canvases[0].bbox_value,
                (200, -40, 200, -40),
            )
        )
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, 1920)
        self.assertLessEqual(bottom, 1080)

    def test_runtime_hud_can_be_hidden_without_hiding_detection_boxes(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(1, 2, 30, 40, 0.8, 0),),
            None,
            1920,
            1080,
        )
        try:
            style = overlay_module.OverlayStyle(hud_visible=False)
        except TypeError as exc:
            self.fail(f"OverlayStyle must accept hud_visible: {exc}")

        overlay.render(
            frame,
            now=10.0,
            runtime=("120 FPS", "DirectML", "1.0Ã—"),
            style=style,
        )

        self.assertEqual(len(canvases[0].items), 1)
        self.assertEqual(canvases[0].items[0][1]["tags"], ("detection",))

    def test_runtime_hud_uses_independent_text_color_and_font_size(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        try:
            style = overlay_module.OverlayStyle(
                box_color="#ff2b2b",
                hud_color="#00cc88",
                hud_font_size=18,
            )
        except TypeError as exc:
            self.fail(f"OverlayStyle must accept HUD typography: {exc}")

        overlay.render(
            None,
            now=10.0,
            runtime=("120 FPS", "DirectML", "1.0Ã—"),
            style=style,
        )

        hud = canvases[0].items[-1][1]
        self.assertEqual(hud["fill"], "#00cc88")
        self.assertEqual(hud["font"], ("Consolas", 18, "bold"))

    def test_runtime_hud_metrics_can_be_enabled_independently(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(1, 2, 30, 40, 0.8, 0),),
            0,
            1920,
            1080,
        )
        try:
            style = overlay_module.OverlayStyle(
                hud_show_fps=False,
                hud_show_provider=True,
                hud_show_zoom=False,
                hud_show_lock=True,
            )
        except TypeError as exc:
            self.fail(f"OverlayStyle must accept HUD metric toggles: {exc}")

        overlay.render(
            frame,
            now=10.0,
            runtime=("120 FPS", "DirectML", "1.5Ã—"),
            style=style,
        )

        self.assertEqual(
            canvases[0].items[-1][1]["text"],
            "AI RUNTIME\nPROVIDER: DirectML\nLOCK: PLAYER",
        )

    def test_runtime_hud_maps_only_supported_selected_classes(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        cases = (
            (Detection(1, 2, 30, 40, 0.8, 0), 0, "PLAYER"),
            (Detection(1, 2, 30, 40, 0.8, 7), 0, "HEAD"),
            (Detection(1, 2, 30, 40, 0.8, 99), 0, "NONE"),
            (Detection(1, 2, 30, 40, 0.8, 0), None, "NONE"),
            (Detection(1, 2, 30, 40, 0.8, 0), 2, "NONE"),
        )

        for detection, selected_index, expected in cases:
            with self.subTest(
                class_id=detection.class_id,
                selected_index=selected_index,
            ):
                frame = DetectionFrameSnapshot(
                    1,
                    10.0,
                    (detection,),
                    selected_index,
                    1920,
                    1080,
                )
                overlay.render(
                    frame,
                    now=10.0,
                    runtime=("120 FPS", "DirectML", "1.0×"),
                )
                self.assertIn(
                    f"LOCK: {expected}",
                    canvases[0].items[-1][1]["text"],
                )

    def test_hidden_head_box_still_reports_selected_head_lock(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(100, 110, 130, 150, 0.9, 7),),
            0,
            1920,
            1080,
        )

        overlay.render(
            frame,
            now=10.0,
            show_heads=False,
            runtime=("120 FPS", "DirectML", "1.0×"),
        )

        self.assertEqual(len(canvases[0].items), 1)
        self.assertEqual(canvases[0].items[0][1]["tags"], ("runtime",))
        self.assertIn("LOCK: HEAD", canvases[0].items[0][1]["text"])

    def test_runtime_hud_drops_lock_when_detection_frame_is_stale(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(100, 110, 130, 150, 0.9, 0),),
            0,
            1920,
            1080,
        )

        try:
            overlay.render(
                frame,
                now=10.151,
                runtime=("0 FPS", "No provider", "1.0×"),
            )
        except TypeError as exc:
            self.fail(f"render must accept runtime status: {exc}")

        self.assertEqual(len(canvases[0].items), 1)
        self.assertIn("LOCK: NONE", canvases[0].items[0][1]["text"])

    def test_render_replaces_runtime_hud_instead_of_accumulating_text(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()

        try:
            overlay.render(
                None,
                now=10.0,
                runtime=("10 FPS", "DirectML", "1.0×"),
            )
            overlay.render(
                None,
                now=10.1,
                runtime=("20 FPS", "DirectML", "1.0×"),
            )
        except TypeError as exc:
            self.fail(f"render must accept runtime status: {exc}")

        runtime_items = [
            item
            for item in canvases[0].items
            if "runtime" in item[1].get("tags", ())
        ]
        self.assertEqual(len(runtime_items), 1)
        self.assertIn("FPS: 20 FPS", runtime_items[0][1]["text"])

    def test_render_keeps_transparency_key_distinct_from_requested_color(self):
        overlay, _window, canvases, _adapter, calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(1, 2, 30, 40, 0.8, 0),),
            None,
            1920,
            1080,
        )

        overlay.render(frame, now=10.0, color="#010203")

        self.assertEqual(canvases[0].items[0][1]["outline"], "#010203")
        self.assertEqual(canvases[0].options["background"], "#010204")
        self.assertIn(("-transparentcolor", "#010204"), calls)

    def test_render_keeps_transparency_key_distinct_from_hud_color(self):
        overlay, _window, canvases, _adapter, calls = self.make_overlay()
        overlay.show()

        overlay.render(
            None,
            now=10.0,
            runtime=("120 FPS", "DirectML", "1.0Ã—"),
            style=overlay_module.OverlayStyle(hud_color="#010203"),
        )

        self.assertEqual(canvases[0].items[-1][1]["fill"], "#010203")
        self.assertEqual(canvases[0].options["background"], "#010204")
        self.assertIn(("-transparentcolor", "#010204"), calls)

    def test_stale_render_clears_existing_boxes_while_window_stays_visible(self):
        overlay, _window, canvases, _adapter, _calls = self.make_overlay()
        overlay.show()
        frame = DetectionFrameSnapshot(
            1,
            10.0,
            (Detection(1, 2, 30, 40, 0.8, 0),),
            None,
            1920,
            1080,
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

        self.assertEqual(
            calls[:3],
            [("delete", "detection"), ("delete", "runtime"), "withdraw"],
        )
        self.assertEqual(adapter.handles, [1234])
        self.assertTrue(overlay.visible)

    def test_close_clears_then_destroys_once_and_prevents_reopening(self):
        overlay, window, _canvases, _adapter, calls = self.make_overlay()
        overlay.show()
        calls.clear()

        overlay.close()
        overlay.close()

        self.assertEqual(
            calls,
            [("delete", "detection"), ("delete", "runtime"), "destroy"],
        )
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

        with self.assertLogs("jitter_app.presentation.overlay", level="ERROR") as logged:
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
