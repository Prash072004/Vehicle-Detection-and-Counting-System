"""Run with: myvenv\Scripts\python.exe -B -m unittest discover -s tests -v.

Model/OCR calls are mocked. OpenCV decoding and SORT also get small real tests.
No inference downloads, bytecode caches, or environment changes are needed.
"""
from fastapi.testclient import TestClient

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as application


def result(rows):
    return [SimpleNamespace(boxes=SimpleNamespace(data=np.asarray(rows, dtype=float)))]


class Capture:
    def __init__(self, frames, frame_count=None):
        self.frames = list(frames)
        self.frame_count = len(frames) if frame_count is None else frame_count
        self.position = 0
        self.read_calls = 0
        self.release = Mock()
        self.isOpened = Mock(return_value=True)

    def read(self):
        self.read_calls += 1
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        if isinstance(frame, Exception):
            raise frame
        return frame is not None, frame

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        return self.position


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        # Existing model/route tests must never create production storage.
        storage_patch = patch.dict(os.environ, {'TRAFFIC_STORAGE_ENABLED': 'false'})
        storage_patch.start()
        self.addCleanup(storage_patch.stop)
        self.logger_patch = patch.object(application, 'logger')
        self.logger = self.logger_patch.start()
        self.addCleanup(self.logger_patch.stop)
        application.cap = None
        application.coco_model = None
        application.license_plate_detector_model = None
        application.mot_tracker = None
        application._initialized = False
        application._processing_finished = False
        application._initialization_error = None
        application.crossing_detector = application.CrossingDetector(application.CROSSING_CONFIG)
        application._frame_index = -1
        application._last_video_timestamp_ms = None
        application._signal_version = 0
        application.traffic_light = 'green'
        application.plate_memory.clear()
        application.app = application.create_app()
        self.addCleanup(application.app.state.processing.stop)
        self.frame = np.zeros((480, 720, 3), dtype=np.uint8)
        self.addCleanup(application.shutdown_resources)

    def prepare_stream(self, frames=None, detections=None, tracks=None):
        # Each replacement capture represents an independent synthetic session.
        application.crossing_detector = application.CrossingDetector(application.CROSSING_CONFIG)
        application._frame_index = -1
        application._last_video_timestamp_ms = None
        capture = Capture([self.frame.copy()] if frames is None else frames)
        application.cap = capture
        application._initialized = True
        application.coco_model = Mock(return_value=result([] if detections is None else detections))
        application.license_plate_detector_model = Mock(return_value=result([]))
        application.mot_tracker = Mock()
        application.mot_tracker.update.return_value = (
            np.empty((0, 5)) if tracks is None else np.asarray(tracks, dtype=float)
        )
        application.get_car = Mock(return_value=(260, 20, 340, 400, 7))
        application.read_license_plate = Mock(return_value=('AB12CDE', 0.9))
        return capture

    def dependencies(self):
        sort_module = ModuleType('sort.sort')
        sort_module.Sort = Mock(return_value=Mock())
        util_module = ModuleType('util')
        util_module.get_car = Mock()
        util_module.read_license_plate = Mock()
        return {'sort.sort': sort_module, 'util': util_module}

    def prepare_plate(self):
        capture = self.prepare_stream(tracks=[[260, 20, 340, 400, 7]])
        application.traffic_light = 'red'
        # Seed an actual preceding approach: Phase 2 rejects first sightings
        # already below the line. Keep every original stability assertion.
        application.crossing_detector.update(7, (300, 310), 0, 'red')
        application._frame_index = 0
        application.license_plate_detector_model.return_value = result([[280, 350, 310, 370, 0.9, 0]])
        return capture

    def test_paths_are_project_relative_and_assets_exist(self):
        self.assertEqual(application.VIDEO_PATH, ROOT / 'video/sample.mp4')
        for path in (application.VIDEO_PATH, application.VEHICLE_MODEL_PATH, application.PLATE_MODEL_PATH):
            self.assertTrue(path.is_file(), str(path))

    def test_initialization_is_idempotent_and_uses_absolute_paths(self):
        capture = Capture([])
        with patch.object(application.cv2, 'VideoCapture', return_value=capture) as open_video, \
                patch.object(application, '_load_yolo_model', side_effect=[Mock(), Mock()]) as load_model, \
                patch.dict(sys.modules, self.dependencies()):
            application.initialize_resources()
            application.initialize_resources()
        open_video.assert_called_once_with(str(ROOT / 'video/sample.mp4'))
        self.assertEqual(load_model.call_count, 2)
        self.assertIs(application.cap, capture)
        capture.release.assert_not_called()

    def test_missing_video_fails_before_capture_or_models(self):
        with patch.object(application, 'VIDEO_PATH', ROOT / 'video/__missing__.mp4'), \
                patch.object(application.cv2, 'VideoCapture') as open_video, \
                patch.object(application, '_load_yolo_model') as load_model:
            with self.assertRaisesRegex(RuntimeError, 'Video source.*missing'):
                application.initialize_resources()
        open_video.assert_not_called()
        load_model.assert_not_called()

    def test_missing_either_weight_fails_before_capture_or_loading(self):
        for name in ('VEHICLE_MODEL_PATH', 'PLATE_MODEL_PATH'):
            with self.subTest(model=name):
                application._initialization_error = None
                with patch.object(application, name, ROOT / 'models/__missing__.pt'), \
                        patch.object(application.cv2, 'VideoCapture') as open_video, \
                        patch.object(application, '_load_yolo_model') as load_model:
                    with self.assertRaisesRegex(RuntimeError, 'YOLO weight file.*missing'):
                        application.initialize_resources()
                open_video.assert_not_called()
                load_model.assert_not_called()

    def test_unopenable_video_is_released_and_error_is_not_retried(self):
        capture = Capture([])
        capture.isOpened.return_value = False
        with patch.object(application.cv2, 'VideoCapture', return_value=capture) as open_video:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, 'codec support'):
                    application.initialize_resources()
        open_video.assert_called_once()
        capture.release.assert_called_once()

    def test_capture_constructor_exception_is_wrapped(self):
        with patch.object(application.cv2, 'VideoCapture', side_effect=cv2.error('unsupported codec')):
            with self.assertRaisesRegex(RuntimeError, 'Could not open video'):
                application.initialize_resources()

    def test_model_failure_releases_capture_without_partial_publication(self):
        capture = Capture([])
        with patch.object(application.cv2, 'VideoCapture', return_value=capture), \
                patch.object(application, '_load_yolo_model', side_effect=[Mock(), RuntimeError('bad weights')]):
            with self.assertRaisesRegex(RuntimeError, 'bad weights'):
                application.initialize_resources()
        self.assertIsNone(application.cap)
        self.assertIsNone(application.coco_model)
        capture.release.assert_called_once()

    def test_model_loader_wraps_constructor_error_without_alternative_weights(self):
        module = ModuleType('ultralytics')
        module.YOLO = Mock(side_effect=RuntimeError('invalid checkpoint'))
        with patch.dict(sys.modules, {'ultralytics': module}):
            with self.assertRaisesRegex(RuntimeError, 'Could not load YOLO model.*invalid checkpoint'):
                application._load_yolo_model(application.VEHICLE_MODEL_PATH)
        module.YOLO.assert_called_once_with(str(application.VEHICLE_MODEL_PATH))

    def test_ocr_initialization_failure_releases_capture(self):
        dependencies = self.dependencies()
        del dependencies['util'].read_license_plate
        capture = Capture([])
        with patch.object(application.cv2, 'VideoCapture', return_value=capture), \
                patch.object(application, '_load_yolo_model', return_value=Mock()), \
                patch.dict(sys.modules, dependencies):
            with self.assertRaisesRegex(RuntimeError, 'Video initialization failed'):
                application.initialize_resources()
        capture.release.assert_called_once()

    def test_eof_stops_before_resize_or_encoding_and_does_not_restart(self):
        capture = self.prepare_stream([])
        capture.frame_count = 1
        capture.position = 1
        with patch.object(application.cv2, 'resize') as resize, \
                patch.object(application.cv2, 'imencode') as encode:
            self.assertEqual(list(application.generate_frames()), [])
            self.assertEqual(list(application.generate_frames()), [])
        resize.assert_not_called()
        encode.assert_not_called()
        capture.release.assert_called_once()
        self.assertEqual(capture.read_calls, 1)
        self.logger.info.assert_called_with('Video processing completed: %s', application.VIDEO_PATH)

    def test_unreadable_frame_stops_without_stale_output_or_retry(self):
        capture = self.prepare_stream([self.frame, None, self.frame])
        chunks = list(application.generate_frames())
        self.assertEqual(len(chunks), 1)
        self.assertEqual(capture.read_calls, 2)
        capture.release.assert_called_once()
        self.logger.warning.assert_called_once()

    def test_empty_frame_never_reaches_resize(self):
        capture = self.prepare_stream([np.empty((0, 0, 3), dtype=np.uint8)])
        with patch.object(application.cv2, 'resize') as resize:
            self.assertEqual(list(application.generate_frames()), [])
        resize.assert_not_called()
        capture.release.assert_called_once()

    def test_read_exception_stops_and_logs_stage(self):
        capture = self.prepare_stream([cv2.error('read failure')])
        self.assertEqual(list(application.generate_frames()), [])
        self.assertEqual(self.logger.exception.call_args[0][-1], 'frame reading')
        capture.release.assert_called_once()

    def test_zero_one_and_multiple_detection_arrays(self):
        rows = [[10, 20, 100, 200, 0.9, 2], [200, 20, 300, 200, 0.8, 7]]
        for count in (0, 1, 2):
            with self.subTest(count=count):
                application._processing_finished = False
                self.prepare_stream(detections=rows[:count])
                self.assertEqual(len(list(application.generate_frames())), 1)
                values = application.mot_tracker.update.call_args[0][0]
                self.assertEqual(values.shape, (count, 5))
                self.assertEqual(values.dtype, np.dtype(float))
                np.testing.assert_array_equal(values, np.asarray(rows[:count], dtype=float)[:, :5]
                                              if count else np.empty((0, 5)))

    def test_non_vehicle_class_filter_is_preserved(self):
        self.prepare_stream(detections=[[10, 20, 100, 200, 0.9, 0]])
        list(application.generate_frames())
        self.assertEqual(application.mot_tracker.update.call_args[0][0].shape, (0, 5))

    def test_consecutive_frames_without_vehicles(self):
        self.prepare_stream([self.frame.copy() for _ in range(4)])
        self.assertEqual(len(list(application.generate_frames())), 4)
        for call in application.mot_tracker.update.call_args_list:
            self.assertEqual(call[0][0].shape, (0, 5))

    def test_output_resolution_and_mjpeg_boundary_are_preserved(self):
        self.prepare_stream([np.zeros((24, 32, 3), dtype=np.uint8)])
        chunks = list(application.generate_frames())
        self.assertTrue(chunks[0].startswith(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'))
        self.assertTrue(chunks[0].endswith(b'\r\n'))
        jpeg = chunks[0].split(b'\r\n\r\n', 1)[1][:-2]
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(frame.shape, (480, 720, 3))

    def test_jpeg_failure_skips_frame_and_never_reads_bad_buffer(self):
        self.prepare_stream([self.frame.copy(), self.frame.copy()])
        failed_buffer = Mock()
        good_buffer = np.frombuffer(b'jpeg', dtype=np.uint8)
        with patch.object(application.cv2, 'imencode', side_effect=[(False, failed_buffer), (True, good_buffer)]):
            chunks = list(application.generate_frames())
        self.assertEqual(len(chunks), 1)
        failed_buffer.tobytes.assert_not_called()

    def test_jpeg_exception_can_recover_on_next_frame(self):
        self.prepare_stream([self.frame.copy(), self.frame.copy()])
        with patch.object(application.cv2, 'imencode', side_effect=[cv2.error('encode'), (True, np.array([1]))]):
            self.assertEqual(len(list(application.generate_frames())), 1)

    def test_repeated_jpeg_failure_is_bounded(self):
        capture = self.prepare_stream([self.frame.copy() for _ in range(10)])
        with patch.object(application.cv2, 'imencode', return_value=(False, None)) as encode:
            self.assertEqual(list(application.generate_frames()), [])
        self.assertEqual(encode.call_count, application.MAX_ENCODING_FAILURES)
        self.assertEqual(capture.read_calls, application.MAX_ENCODING_FAILURES)
        capture.release.assert_called_once()

    def test_resize_inference_tracking_and_annotation_failures_stop(self):
        for stage in ('frame resizing', 'vehicle inference', 'SORT tracking', 'frame annotation'):
            with self.subTest(stage=stage):
                application._processing_finished = False
                capture = self.prepare_stream()
                targets = {
                    'frame resizing': (application.cv2, 'resize'),
                    'vehicle inference': (application, 'coco_model'),
                    'SORT tracking': (application.mot_tracker, 'update'),
                    'frame annotation': (application.cv2, 'line'),
                }
                with patch.object(*targets[stage], side_effect=RuntimeError('permanent failure')):
                    self.assertEqual(list(application.generate_frames()), [])
                self.assertEqual(self.logger.exception.call_args[0][-1], stage)
                capture.release.assert_called_once()
                self.assertEqual(capture.read_calls, 1)

    def test_red_highlight_plate_detection_and_ocr_are_preserved(self):
        self.prepare_plate()
        with patch.object(application.cvzone, 'cornerRect') as draw:
            self.assertEqual(len(list(application.generate_frames())), 1)
        self.assertEqual(draw.call_args[1]['colorR'], (0, 0, 255))
        self.assertEqual(application.plate_memory, {7: 'AB12CDE'})
        application.license_plate_detector_model.assert_called_once()
        application.read_license_plate.assert_called_once()

    def test_green_does_not_run_plate_detection(self):
        self.prepare_stream(tracks=[[10, 20, 100, 400, 7]])
        with patch.object(application.cvzone, 'cornerRect') as draw:
            list(application.generate_frames())
        self.assertEqual(draw.call_args[1]['colorR'], (0, 255, 0))
        application.license_plate_detector_model.assert_not_called()

    def test_plate_inference_and_ocr_failures_stop_with_stage(self):
        for attribute, stage in (('license_plate_detector_model', 'license plate inference'),
                                 ('read_license_plate', 'OCR processing')):
            with self.subTest(stage=stage):
                application._processing_finished = False
                capture = self.prepare_plate()
                with patch.object(application, attribute, side_effect=RuntimeError('failure')):
                    self.assertEqual(list(application.generate_frames()), [])
                self.assertEqual(self.logger.exception.call_args[0][-1], stage)
                capture.release.assert_called_once()

    def test_empty_plate_crop_stops_before_ocr(self):
        capture = self.prepare_plate()
        application.license_plate_detector_model.return_value = result([[30, 350, 30, 370, 0.9, 0]])
        self.assertEqual(list(application.generate_frames()), [])
        application.read_license_plate.assert_not_called()
        capture.release.assert_called_once()

    def test_disconnect_does_not_release_capture_used_by_another_generator(self):
        capture = self.prepare_stream([self.frame.copy() for _ in range(3)])
        first = application.generate_frames()
        second = application.generate_frames()
        next(first)
        next(second)
        first.close()
        capture.release.assert_not_called()
        self.assertEqual(len(list(second)), 1)
        capture.release.assert_called_once()

    def test_shutdown_is_idempotent_and_cleanup_errors_do_not_escape(self):
        capture = self.prepare_stream()
        capture.release.side_effect = RuntimeError('release failed')
        application.shutdown_resources()
        application.shutdown_resources()
        capture.release.assert_called_once()
        self.assertIsNone(application.cap)
        self.logger.exception.assert_called_with('Could not release video capture')

    def test_routes_remain_compatible_and_usable_after_eof(self):
        self.prepare_stream()
        client = TestClient(application.app, follow_redirects=False)
        self.assertEqual(client.get('/').status_code, 200)
        response = client.post('/set_light/red')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], '/')
        self.assertEqual(application.traffic_light, 'red')
        self.assertEqual(client.get('/violations').json(), [])
        response = client.get('/video')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['content-type'], 'multipart/x-mixed-replace; boundary=frame')
        self.assertTrue(response.content.startswith(b'--frame\r\n'))
        application.plate_memory[7] = 'AB12CDE'
        self.assertEqual(client.get('/violations').json(), ['AB12CDE'])
        self.assertEqual(client.get('/').status_code, 200)
        self.assertEqual(client.post('/set_light/green').status_code, 302)
        client.post('/set_light/yellow')
        self.assertEqual(application.traffic_light, 'green')
        self.assertEqual(client.get('/video').content, b'')

    def test_video_startup_failure_returns_503_without_breaking_other_routes(self):
        with patch.object(application, 'VIDEO_PATH', ROOT / 'video/__missing__.mp4'):
            client = TestClient(application.app, follow_redirects=False)
            self.assertEqual(client.get('/video').status_code, 503)
            self.assertEqual(client.get('/').status_code, 200)
            self.assertEqual(client.get('/violations').json(), [])

    def test_direct_startup_disables_debug_reloader_and_always_cleans_up(self):
        with patch.object(application.uvicorn, 'run') as run, \
                patch.object(application.app.state.processing, 'stop') as stop:
            self.assertEqual(application.main(), 0)
        run.assert_called_once_with(application.app, host='127.0.0.1', port=8000,
                                    workers=1, reload=False, timeout_graceful_shutdown=5)
        stop.assert_called_once()

    def test_startup_failure_returns_nonzero_without_starting_server(self):
        with patch.object(application.uvicorn, 'run', side_effect=RuntimeError('startup')) as run, \
                patch.object(application.app.state.processing, 'stop') as stop:
            self.assertEqual(application.main(), 1)
        run.assert_called_once()
        stop.assert_called_once()

    def test_import_does_not_open_video_or_initialize_models(self):
        specification = importlib.util.spec_from_file_location('stability_import_check', ROOT / 'main.py')
        module = importlib.util.module_from_spec(specification)
        with patch.object(cv2, 'VideoCapture') as open_video:
            specification.loader.exec_module(module)
        open_video.assert_not_called()
        self.assertIsNone(module.coco_model)
        self.assertIsNone(module.license_plate_detector_model)
        self.assertFalse(module._initialized)


class RealComponentTests(unittest.TestCase):
    def test_default_video_decodes_resizes_and_encodes(self):
        capture = cv2.VideoCapture(str(ROOT / 'video/sample.mp4'))
        try:
            self.assertTrue(capture.isOpened())
            ok, frame = capture.read()
            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            frame = cv2.resize(frame, (720, 480))
            ok, buffer = cv2.imencode('.jpg', frame)
            self.assertTrue(ok)
            self.assertGreater(buffer.size, 0)
        finally:
            capture.release()

    def test_actual_sort_handles_disappearing_vehicles_and_empty_frames(self):
        with tempfile.TemporaryDirectory() as config_directory:
            with patch.dict(os.environ, {'MPLCONFIGDIR': config_directory}):
                from sort.sort import Sort
                tracker = Sort()
                self.assertEqual((tracker.max_age, tracker.min_hits, tracker.iou_threshold), (1, 3, 0.3))
                self.assertEqual(tracker.update(np.empty((0, 5))).shape, (0, 5))
                one = np.asarray([[10, 10, 40, 40, 0.9]], dtype=float)
                tracked = tracker.update(one)
                self.assertEqual(tracked.shape, (1, 5))
                identifier = tracked[0, 4]
                self.assertEqual(tracker.update(one)[0, 4], identifier)
                two = np.vstack([one, [100, 100, 150, 150, 0.8]])
                for _ in range(4):
                    tracked = tracker.update(two)
                self.assertEqual(tracked.shape, (2, 5))
                self.assertEqual(len(set(tracked[:, 4])), 2)
                for _ in range(3):
                    self.assertEqual(tracker.update(np.empty((0, 5))).shape, (0, 5))
                self.assertEqual(tracker.trackers, [])


if __name__ == '__main__':
    unittest.main()
