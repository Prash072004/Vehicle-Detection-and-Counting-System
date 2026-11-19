"""Phase 3 detector boundary, configuration and application regressions."""
from fastapi.testclient import TestClient

from dataclasses import replace
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vehicle_detector import (DetectionConfig, DetectionCounts, RegionOfInterest,
                              detect_vehicles, prepare_detections, resolve_video_source)


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.config = DetectionConfig()
        self.row = [10, 20, 100, 200, 0.9, 2]

    def convert(self, rows, config=None, size=(720, 480), counts=None):
        return prepare_detections(rows, size, config or self.config, counts)

    def test_supported_classes_and_unsupported_objects(self):
        rows = [self.row[:5] + [class_id] for class_id in (0, 1, 2, 3, 5, 7, 9)]
        self.assertEqual(self.convert(rows).shape, (4, 5))

    def test_confidence_cutoff_is_inclusive_at_adapter_boundary(self):
        rows = [self.row[:4] + [score, 2] for score in (0.249, 0.25, 0.251)]
        np.testing.assert_array_equal(self.convert(rows)[:, 4], [0.25, 0.251])

    def test_configured_cutoff_and_probability_endpoints(self):
        for cutoff in (0, 0.6, 1):
            with self.subTest(cutoff=cutoff):
                rows = [self.row[:4] + [score, 2] for score in (0, 0.6, 1)]
                self.assertEqual(len(self.convert(rows, DetectionConfig(confidence=cutoff))),
                                 sum(score >= cutoff for score in (0, 0.6, 1)))

    def test_invalid_confidence_settings_fail(self):
        for value in (-0.1, 1.1, float('nan'), float('inf'), True, '0.25', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DetectionConfig(confidence=value)

    def test_invalid_detection_confidences_never_reach_sort(self):
        rows = [self.row[:4] + [value, 2] for value in
                (-1, 2, float('nan'), float('inf'), True, '0.9', None)]
        self.assertEqual(self.convert(rows).shape, (0, 5))

    def test_invalid_coordinates_and_malformed_rows_are_skipped(self):
        rows = [None, [], [1, 2], self.row + [9]]
        for value in (float('nan'), float('inf'), -float('inf'), '10', None, True):
            for index in range(4):
                row = self.row.copy()
                row[index] = value
                rows.append(row)
        self.assertEqual(self.convert(rows).shape, (0, 5))

    def test_zero_and_reversed_areas_are_rejected(self):
        rows = [[10, 20, 10, 200, .9, 2], [10, 20, 100, 20, .9, 2],
                [100, 20, 10, 200, .9, 2], [10, 200, 100, 20, .9, 2]]
        self.assertEqual(self.convert(rows).shape, (0, 5))

    def test_negative_and_partial_boxes_clip_to_actual_frame(self):
        rows = [[-10, -20, 800, 500, .9, 2], [700, 450, 900, 600, .8, 7]]
        np.testing.assert_array_equal(self.convert(rows),
            [[0, 0, 720, 480, .9], [700, 450, 720, 480, .8]])

    def test_completely_outside_boxes_are_rejected_after_clipping(self):
        rows = [[-100, 20, -10, 200, .9, 2], [730, 20, 800, 200, .9, 2],
                [10, -20, 100, -1, .9, 2], [10, 490, 100, 500, .9, 2]]
        self.assertEqual(self.convert(rows).shape, (0, 5))

    def test_noninteger_or_nonfinite_class_ids_are_not_truncated(self):
        rows = [self.row[:5] + [value] for value in (2.9, float('nan'), float('inf'), True, None)]
        self.assertEqual(self.convert(rows).shape, (0, 5))

    def test_empty_output_is_float_zero_by_five(self):
        values = self.convert([])
        self.assertEqual(values.shape, (0, 5))
        self.assertEqual(values.dtype, np.dtype(float))

    def test_mixed_rows_and_rejection_counts(self):
        counts = DetectionCounts()
        rows = [self.row, [0, 0, 0, 2, .9, 2], self.row[:4] + [.1, 2], self.row[:5] + [0]]
        np.testing.assert_array_equal(self.convert(rows, counts=counts), [self.row[:5]])
        self.assertEqual((counts.returned_by_yolo, counts.invalid, counts.below_confidence,
                          counts.unsupported_class, counts.accepted), (4, 1, 1, 1, 1))

    def test_input_rows_are_not_modified(self):
        rows = [[-10, 20, 800, 500, .9, 2]]
        original = [row.copy() for row in rows]
        self.convert(rows)
        self.assertEqual(rows, original)

    def test_invalid_frame_dimensions_are_rejected(self):
        for size in (None, (0, 480), (720, -1), (720., 480), (True, 480), (720,)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.convert([], size=size)

    def test_configurable_inference_size_preserves_processing_coordinates(self):
        from test_stability import result
        model = Mock(return_value=result([self.row]))
        frame = np.zeros((480, 720, 3), dtype=np.uint8)
        config = DetectionConfig(inference_size=1280, device='cpu')
        values = detect_vehicles(model, frame, config)
        self.assertIs(model.call_args[0][0], frame)
        self.assertEqual(model.call_args[1], {'conf': .25, 'imgsz': 1280, 'iou': .7, 'device': 'cpu'})
        np.testing.assert_array_equal(values, [self.row[:5]])

    def test_invalid_inference_size_and_device(self):
        for size in (0, -32, 641, 16, True, 640., '640'):
            with self.subTest(size=size), self.assertRaises(ValueError):
                DetectionConfig(inference_size=size)
        for device in ('', 'gpu', -1, ['cpu']):
            with self.subTest(device=device), self.assertRaises(ValueError):
                DetectionConfig(device=device)


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.roi = RegionOfInterest(enabled=True, bounds=(100, 100, 300, 300))

    def test_inside_outside_and_all_rectangle_boundaries(self):
        for point, expected in [((200, 200), True), ((100, 200), True), ((300, 200), True),
                                ((200, 100), True), ((200, 300), True), ((99, 200), False),
                                ((301, 200), False), ((200, 301), False)]:
            x, y = point
            with self.subTest(point=point):
                self.assertEqual(self.roi.contains((x - 10, y - 20, x + 10, y)), expected)

    def test_partial_overlap_uses_representative_point(self):
        self.assertTrue(self.roi.contains((50, 50, 250, 250)))
        self.assertFalse(self.roi.contains((50, 50, 110, 110)))

    def test_center_anchor_can_differ_from_bottom_center(self):
        box = (150, 150, 250, 350)
        self.assertFalse(self.roi.contains(box))
        self.assertTrue(replace(self.roi, anchor='center').contains(box))

    def test_disabled_roi_preserves_all_valid_detections(self):
        roi = replace(self.roi, enabled=False)
        row = [400, 300, 500, 450, .9, 2]
        self.assertEqual(len(prepare_detections([row], (720, 480), DetectionConfig(roi=roi))), 1)

    def test_roi_filtering_and_counts_use_clipped_boxes(self):
        counts = DetectionCounts()
        roi = RegionOfInterest(True, (0, 0, 100, 100))
        config = DetectionConfig(roi=roi)
        values = prepare_detections([[-100, -100, 100, 100, .9, 2],
                                     [200, 200, 300, 300, .8, 2]], (720, 480), config, counts)
        np.testing.assert_array_equal(values, [[0, 0, 100, 100, .9]])
        self.assertEqual(counts.outside_roi, 1)

    def test_roi_frame_mismatch_fails_even_without_detections(self):
        with self.assertRaisesRegex(ValueError, 'reference size'):
            prepare_detections([], (1280, 720), DetectionConfig(roi=self.roi))

    def test_invalid_roi_settings_fail(self):
        values = [dict(enabled=True), dict(enabled=1), dict(anchor='crossing'),
                  dict(bounds=(1, 2)), dict(bounds=1), dict(bounds=(1, 2, 1, 4)),
                  dict(bounds=(-1, 0, 100, 100)), dict(bounds=(0, 0, 721, 480)),
                  dict(bounds=(0, 0, float('nan'), 20)), dict(reference_size=(720, 0))]
        for kwargs in values:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RegionOfInterest(**kwargs)


class ConfigurationTests(unittest.TestCase):
    def test_default_environment_preserves_original_detection_settings(self):
        config = DetectionConfig.from_environment(environ={})
        self.assertEqual(config.inference_arguments(), {'conf': .25, 'imgsz': 640, 'iou': .7})
        self.assertFalse(config.roi.enabled)

    def test_environment_configuration(self):
        config = DetectionConfig.from_environment(environ={
            'TRAFFIC_VEHICLE_CONFIDENCE': '.45', 'TRAFFIC_VEHICLE_IMGSZ': '960',
            'TRAFFIC_VEHICLE_DEVICE': 'cpu', 'TRAFFIC_ROI_ENABLED': 'true',
            'TRAFFIC_ROI': '100,100,500,400', 'TRAFFIC_ROI_ANCHOR': 'center'})
        self.assertEqual(config.confidence, .45)
        self.assertEqual(config.inference_size, 960)
        self.assertTrue(config.roi.enabled)
        self.assertEqual(config.roi.bounds, (100, 100, 500, 400))
        self.assertEqual(config.roi.anchor, 'center')

    def test_bounds_alone_do_not_enable_roi(self):
        config = DetectionConfig.from_environment(environ={'TRAFFIC_ROI': '0,0,500,400'})
        self.assertFalse(config.roi.enabled)

    def test_invalid_environment_settings_fail(self):
        for env in ({'TRAFFIC_VEHICLE_CONFIDENCE': 'bad'}, {'TRAFFIC_VEHICLE_IMGSZ': '640.5'},
                    {'TRAFFIC_ROI_ENABLED': 'yes'}, {'TRAFFIC_ROI': 'bad'},
                    {'TRAFFIC_ROI_ENABLED': '1'}, {'TRAFFIC_VEHICLE_DEVICE': ''}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                DetectionConfig.from_environment(environ=env)

    def test_local_paths_resolve_independently_of_working_directory(self):
        self.assertEqual(resolve_video_source('video/sample.mp4', ROOT), ROOT / 'video/sample.mp4')
        self.assertEqual(resolve_video_source(ROOT / 'video/sample2.mp4', ROOT), ROOT / 'video/sample2.mp4')

    def test_explicit_webcam_sources_and_invalid_sources(self):
        self.assertEqual(resolve_video_source('webcam:0', ROOT), 0)
        self.assertEqual(resolve_video_source('webcam:2', ROOT), 2)
        for source in ('', 'webcam:-1', 'webcam:a', 'rtsp://example/stream', 'https://example', None, 0):
            with self.subTest(source=source), self.assertRaises(ValueError):
                resolve_video_source(source, ROOT)


class ApplicationDetectionTests(unittest.TestCase):
    def setUp(self):
        import test_stability as stability
        self.stability = stability
        self.app = stability.application
        fixture = stability.ApplicationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.prepare_stream = fixture.prepare_stream
        self.prepare_plate = fixture.prepare_plate
        self.frame = fixture.frame
        config_patch = patch.object(self.app, 'VEHICLE_DETECTION_CONFIG', DetectionConfig())
        config_patch.start()
        self.addCleanup(config_patch.stop)
        source_patch = patch.object(self.app, '_active_video_source', self.app.VIDEO_PATH)
        source_patch.start()
        self.addCleanup(source_patch.stop)
        env_patch = patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        for name in list(os.environ):
            if name.startswith('TRAFFIC_'):
                del os.environ[name]
        self.dependencies = fixture.dependencies

    def test_inference_and_plate_crops_stay_clean_when_display_is_drawn(self):
        self.prepare_plate()
        clean_vehicle_frames, clean_plate_frames = [], []
        def vehicle(frame, **kwargs):
            clean_vehicle_frames.append(frame)
            return self.stability.result([])
        def plate(frame):
            clean_plate_frames.append(frame)
            return self.stability.result([[280, 350, 310, 370, .9, 0]])
        self.app.coco_model.side_effect = vehicle
        self.app.license_plate_detector_model.side_effect = plate
        def annotate(frame, *args, **kwargs):
            frame[:] = 255
        with patch.object(self.app.cv2, 'line', side_effect=annotate):
            chunks = list(self.app.generate_frames())
        self.assertEqual(len(chunks), 1)
        self.assertIs(clean_vehicle_frames[0], clean_plate_frames[0])
        self.assertFalse(clean_plate_frames[0].any())
        # Clean black crops become white under the unchanged inverse threshold.
        np.testing.assert_array_equal(self.app.read_license_plate.call_args[0][0],
                                      np.full((20, 30), 255, dtype=np.uint8))
        jpeg = chunks[0].split(b'\r\n\r\n', 1)[1][:-2]
        display = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertTrue(display[100, 100].any())
        self.assertEqual(self.app.plate_memory, {7: 'AB12CDE'})

    def test_only_valid_filtered_detections_reach_sort(self):
        rows = [[-10, 20, 800, 500, .9, 2], [10, 20, 100, 200, .1, 3],
                [float('nan'), 20, 100, 200, .9, 7], [10, 20, 100, 200, .8, 0]]
        self.prepare_stream(detections=rows)
        self.assertEqual(len(list(self.app.generate_frames())), 1)
        np.testing.assert_array_equal(self.app.mot_tracker.update.call_args[0][0],
                                      [[0, 20, 720, 480, .9]])

    def test_roi_filter_applies_before_tracking(self):
        self.app.VEHICLE_DETECTION_CONFIG = DetectionConfig(roi=RegionOfInterest(True, (0, 0, 100, 250)))
        self.prepare_stream(detections=[[10, 20, 100, 200, .9, 2], [200, 20, 300, 200, .9, 7]])
        self.assertEqual(len(list(self.app.generate_frames())), 1)
        self.assertEqual(self.app.mot_tracker.update.call_args[0][0].shape, (1, 5))

    def test_alternate_local_source_uses_one_capture(self):
        capture = self.stability.Capture([])
        with patch.dict(os.environ, {'TRAFFIC_VIDEO_SOURCE': 'video/sample2.mp4'}), \
                patch.object(self.app.cv2, 'VideoCapture', return_value=capture) as open_video, \
                patch.object(self.app, '_load_yolo_model', return_value=Mock()), \
                patch.dict(sys.modules, self.dependencies()):
            self.app.initialize_resources()
            self.app.initialize_resources()
        open_video.assert_called_once_with(str(ROOT / 'video/sample2.mp4'))

    def test_webcam_source_uses_integer_index(self):
        capture = self.stability.Capture([])
        with patch.dict(os.environ, {'TRAFFIC_VIDEO_SOURCE': 'webcam:0'}), \
                patch.object(self.app.cv2, 'VideoCapture', return_value=capture) as open_video, \
                patch.object(self.app, '_load_yolo_model', return_value=Mock()), \
                patch.dict(sys.modules, self.dependencies()):
            self.app.initialize_resources()
        open_video.assert_called_once_with(0)

    def test_unavailable_webcam_releases_capture_and_reports_503(self):
        capture = self.stability.Capture([])
        capture.isOpened.return_value = False
        with patch.dict(os.environ, {'TRAFFIC_VIDEO_SOURCE': 'webcam:0'}), \
                patch.object(self.app.cv2, 'VideoCapture', return_value=capture):
            self.assertEqual(TestClient(self.app.app, follow_redirects=False).get('/video').status_code, 503)
        capture.release.assert_called_once()

    def test_missing_configured_video_or_model_fails_before_capture(self):
        for env in ({'TRAFFIC_VIDEO_SOURCE': 'video/missing.mp4'},
                    {'TRAFFIC_VEHICLE_MODEL': 'models/missing.pt'}):
            with self.subTest(env=env), patch.dict(os.environ, env), \
                    patch.object(self.app.cv2, 'VideoCapture') as open_video, \
                    patch.object(self.app, '_load_yolo_model') as load:
                self.app._initialization_error = None
                with self.assertRaisesRegex(RuntimeError, 'missing'):
                    self.app.initialize_resources()
                open_video.assert_not_called()
                load.assert_not_called()

    def test_invalid_configuration_fails_before_capture(self):
        with patch.dict(os.environ, {'TRAFFIC_VEHICLE_CONFIDENCE': '2'}), \
                patch.object(self.app.cv2, 'VideoCapture') as open_video:
            with self.assertRaisesRegex(RuntimeError, 'between 0 and 1'):
                self.app.initialize_resources()
            open_video.assert_not_called()

    def test_environment_changes_do_not_switch_an_active_source(self):
        capture = self.prepare_stream()
        with patch.dict(os.environ, {'TRAFFIC_VIDEO_SOURCE': 'video/missing.mp4',
                                     'TRAFFIC_VEHICLE_CONFIDENCE': 'invalid'}), \
                patch.object(self.app.cv2, 'VideoCapture') as open_video:
            self.assertEqual(len(list(self.app.generate_frames())), 1)
            self.assertIsNone(self.app.cap)
            open_video.assert_not_called()
        capture.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
