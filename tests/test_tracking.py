"""Deterministic SORT lifecycle/safety tests; no YOLO weights, OCR, or GUI."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking_manager import (TrackingConfig, TrackingManager, sanitize_tracks, validate_detections)
from sort.sort import Sort


def detection(x=100, y=100, width=80, height=100):
    return np.array([[x, y, x + width, y + height, .9]], dtype=float)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_preserve_sort_values(self):
        manager = TrackingManager()
        self.assertEqual((manager.sort.max_age, manager.sort.min_hits, manager.sort.iou_threshold), (1, 3, .3))

    def test_custom_parameters_reach_sort(self):
        manager = TrackingManager(TrackingConfig(5, 2, .6))
        self.assertEqual((manager.sort.max_age, manager.sort.min_hits, manager.sort.iou_threshold), (5, 2, .6))

    def test_invalid_configuration_type_or_frame_size(self):
        for config in ({}, False, 'default'):
            with self.subTest(config=config), self.assertRaises(ValueError):
                TrackingManager(config)
        with self.assertRaises(ValueError):
            TrackingManager(frame_size=(0, 480))

    def test_invalid_parameters(self):
        for kwargs in ({'max_age': -1}, {'max_age': True}, {'max_age': 1.5},
                       {'min_hits': 0}, {'min_hits': False}, {'min_hits': '3'},
                       {'iou_threshold': -.1}, {'iou_threshold': 1.1},
                       {'iou_threshold': float('nan')}, {'iou_threshold': float('inf')},
                       {'iou_threshold': True}, {'iou_threshold': '.3'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrackingConfig(**kwargs)

    def test_boundary_parameter_values(self):
        self.assertEqual(TrackingConfig(0, 1, 0).max_age, 0)
        self.assertEqual(TrackingConfig(iou_threshold=1).iou_threshold, 1)

    def test_environment_configuration(self):
        config = TrackingConfig.from_environment(environ={'TRAFFIC_SORT_MAX_AGE': '3',
            'TRAFFIC_SORT_MIN_HITS': '2', 'TRAFFIC_SORT_IOU_THRESHOLD': '.4'})
        self.assertEqual(config, TrackingConfig(3, 2, .4))

    def test_invalid_environment_and_no_configuration_mutation(self):
        original = TrackingConfig()
        for env in ({'TRAFFIC_SORT_MAX_AGE': '1.5'}, {'TRAFFIC_SORT_MIN_HITS': '0'},
                    {'TRAFFIC_SORT_IOU_THRESHOLD': 'nan'}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                TrackingConfig.from_environment(original, env)
        self.assertEqual(original, TrackingConfig())


class BoundaryTests(unittest.TestCase):
    def test_detection_shape_empty_and_no_input_mutation(self):
        values = detection()
        original = values.copy()
        self.assertEqual(validate_detections([]).shape, (0, 5))
        self.assertEqual(validate_detections(np.empty((0, 5))).shape, (0, 5))
        np.testing.assert_array_equal(validate_detections(values), original)
        np.testing.assert_array_equal(values, original)

    def test_invalid_input_is_rejected_before_sort_advances(self):
        manager = TrackingManager()
        for rows in (None, [[1, 2]], np.empty((0, 6)), [[1, 2, 1, 3, .9]],
                     [[1, 2, 3, 4, float('nan')]], [[1, 2, float('inf'), 4, .9]],
                     [[1, 2, 3, 4, 2]], [[True, 2, 3, 4, .9]], [['1', 2, 3, 4, .9]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                manager.update(rows)
            self.assertEqual(manager.sort.frame_count, 0)

    def test_clipping_and_rejection_of_completely_outside_input(self):
        np.testing.assert_array_equal(validate_detections([[-10, -10, 800, 600, .9]]), [[0, 0, 720, 480, .9]])
        with self.assertRaises(ValueError):
            validate_detections([[800, 20, 900, 30, .9]])

    def test_valid_output_preserves_positive_identity_and_shape(self):
        rows, rejected = sanitize_tracks([[10, 20, 30, 40, 7]])
        np.testing.assert_array_equal(rows, [[10, 20, 30, 40, 7]])
        self.assertEqual(rejected, set())
        self.assertEqual(sanitize_tracks([])[0].shape, (0, 5))

    def test_duplicate_ids_are_all_rejected(self):
        rows, rejected = sanitize_tracks([[10, 20, 30, 40, 7], [50, 20, 70, 40, 7], [80, 20, 100, 40, 8]])
        self.assertEqual(rows[:, 4].tolist(), [8])
        self.assertEqual(rejected, {7})

    def test_invalid_output_coordinates_or_ids_are_never_emitted(self):
        rows = [[0, 0, float('nan'), 20, 1], [0, 0, 0, 20, 2], [10, 20, 5, 40, 3],
                [800, 0, 900, 20, 4], [0, 0, 10, 20, 0], [0, 0, 10, 20, -1],
                [0, 0, 10, 20, 1.5], [0, 0, 10, 20, float('inf')]]
        safe, rejected = sanitize_tracks(rows)
        self.assertEqual(safe.shape, (0, 5))
        self.assertEqual(rejected, {1, 2, 3, 4})


class LifecycleTests(unittest.TestCase):
    def manager(self, age=1, hits=1):
        return TrackingManager(TrackingConfig(age, hits, .3))

    def confirm(self, manager, frames=2):
        result = None
        for index in range(frames):
            result = manager.update(detection(x=100 + index))
        self.assertEqual(len(result), 1)
        return int(result[0, 4])

    def test_native_startup_bypass_is_suppressed_until_true_confirmation(self):
        manager = TrackingManager()
        for frame in range(3):
            self.assertEqual(manager.update(detection()).shape, (0, 5))
            state = next(iter(manager.tracks.values()))
            self.assertEqual(state.state, 'tentative')
        self.assertEqual(len(manager.update(detection())), 1)
        self.assertTrue(next(iter(manager.tracks.values())).confirmed)

    def test_first_seen_last_observed_and_confirmation_metadata(self):
        manager = self.manager()
        identity = self.confirm(manager)
        state = manager.tracks[identity]
        self.assertEqual((state.first_seen_frame, state.last_observed_frame, state.missed_frames), (0, 1, 0))
        self.assertTrue(state.observed)
        self.assertEqual(state.state, 'confirmed')

    def test_empty_sequence_never_creates_tracks(self):
        manager = self.manager()
        for index in range(10):
            self.assertEqual(manager.update(np.empty((0, 5))).shape, (0, 5))
        self.assertEqual(manager.tracks, {})
        self.assertEqual(manager.created_count, 0)

    def test_one_missed_frame_is_predicted_metadata_only(self):
        manager = self.manager()
        identity = self.confirm(manager)
        result = manager.update([])
        state = manager.tracks[identity]
        self.assertEqual(result.shape, (0, 5))
        self.assertEqual((state.state, state.observed, state.missed_frames), ('lost', False, 1))
        self.assertEqual(state.last_observed_frame, 1)
        self.assertIsNotNone(state.bbox)

    def test_short_gap_recovery_keeps_same_object_and_identity(self):
        manager = self.manager(hits=3)
        identity = self.confirm(manager, frames=4)
        track = manager.sort.trackers[0]
        manager.update([])
        recovered = manager.update(detection(x=105))
        self.assertEqual(int(recovered[0, 4]), identity)
        self.assertIs(manager.sort.trackers[0], track)
        self.assertEqual(track.hit_streak, 1)  # Native SORT would suppress this recovery.
        self.assertEqual(manager.recovered_ids, (identity,))
        self.assertEqual(manager.recovered_count, 1)

    def test_longer_gap_is_retained_only_under_experimental_max_age(self):
        manager = self.manager(age=3)
        identity = self.confirm(manager)
        for missed in range(1, 4):
            self.assertEqual(len(manager.update([])), 0)
            self.assertEqual(manager.tracks[identity].missed_frames, missed)
        recovered = manager.update(detection(x=105))
        self.assertEqual(int(recovered[0, 4]), identity)

    def test_expiration_occurs_on_misses_strictly_greater_than_max_age(self):
        manager = self.manager(age=1)
        identity = self.confirm(manager)
        manager.update([])
        self.assertIn(identity, manager.tracks)
        manager.update([])
        self.assertNotIn(identity, manager.tracks)
        self.assertEqual(manager.expired_ids, (identity,))
        self.assertEqual(manager.expired_count, 1)
        manager.update([])
        self.assertEqual(manager.expired_ids, ())

    def test_zero_age_expires_on_first_miss(self):
        manager = self.manager(age=0)
        identity = self.confirm(manager)
        manager.update([])
        self.assertNotIn(identity, manager.tracks)

    def test_leaving_and_reentering_creates_new_id_without_reusing_old_id(self):
        manager = self.manager(age=0)
        old = self.confirm(manager)
        manager.update([])
        self.assertEqual(len(manager.update(detection())), 0)
        result = manager.update(detection(x=101))
        self.assertNotEqual(int(result[0, 4]), old)
        self.assertEqual(manager.created_count, 2)

    def test_two_separate_vehicles_keep_distinct_ids(self):
        manager = self.manager()
        for index in range(8):
            boxes = np.concatenate((detection(x=100 + index * 3), detection(x=350 - index * 3)))
            result = manager.update(boxes)
            if index == 1:
                identities = set(result[:, 4])
            elif index > 1:
                self.assertEqual(set(result[:, 4]), identities)
        self.assertEqual(len(identities), 2)

    def test_overlapping_crossing_paths_do_not_collapse_tracks(self):
        manager = self.manager(age=3)
        for index in range(15):
            boxes = np.concatenate((detection(x=100 + 8 * index), detection(x=300 - 8 * index)))
            result = manager.update(boxes)
            self.assertTrue(np.isfinite(result).all())
            self.assertEqual(len(result[:, 4]), len(set(result[:, 4])))
        self.assertEqual(len(manager.tracks), 2)
        # Identity correctness through heavy overlap requires labeled evaluation.

    def test_lifecycle_state_is_bounded_after_many_expired_tracks(self):
        manager = self.manager(age=0)
        for index in range(50):
            manager.update(detection())
            manager.update([])
        self.assertEqual(manager.tracks, {})
        self.assertEqual(manager._confirmed, set())
        self.assertEqual(manager.created_count, 50)
        self.assertEqual(manager.expired_count, 50)

    def test_clear_releases_state_without_resetting_id_counter(self):
        manager = self.manager()
        old = self.confirm(manager)
        manager.clear()
        self.assertEqual(manager.sort.trackers, [])
        self.assertEqual(manager.tracks, {})
        self.assertEqual(manager._confirmed, set())
        new = self.confirm(manager)
        self.assertGreater(new, old)

    def test_duplicate_internal_ids_are_ambiguous_even_when_one_is_predicted(self):
        manager = self.manager()
        identity = self.confirm(manager)
        duplicate = manager.sort.trackers[0]
        with patch.object(manager.sort, 'update'), patch.object(manager.sort, 'trackers', [duplicate, duplicate]):
            self.assertEqual(len(manager.update(detection())), 0)
        self.assertIn(identity, manager.invalid_ids)


class NativeSafetyTests(unittest.TestCase):
    def test_infinite_prediction_does_not_shift_matching_to_wrong_tracker(self):
        tracker = Sort(min_hits=1)
        tracker.update(np.concatenate((detection(x=100), detection(x=350))))
        bad, good = tracker.trackers
        with patch.object(bad, 'predict', return_value=np.array([[float('inf'), 100, float('inf'), 200]])):
            rows = tracker.update(detection(x=350))
        self.assertEqual(len(tracker.trackers), 1)
        self.assertIs(tracker.trackers[0], good)
        self.assertEqual(int(rows[0, 4]), good.id + 1)

    def test_zero_area_prediction_is_removed_with_its_association_row(self):
        tracker = Sort(min_hits=1)
        tracker.update(detection())
        bad = tracker.trackers[0]
        with patch.object(bad, 'predict', return_value=np.array([[100, 100, 100, 200]])):
            self.assertEqual(tracker.update([]).shape, (0, 5))
        self.assertEqual(tracker.trackers, [])

    def test_sort_and_flask_import_without_gui_packages_or_rng_reset(self):
        script = '''import importlib.abc,sys,os
os.environ['DISPLAY']=''
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('matplotlib','tkinter','skimage'):
            raise ImportError('GUI import blocked: '+fullname)
sys.meta_path.insert(0,Block())
import numpy as np
np.random.seed(734)
expected=np.random.RandomState(734).rand()
from sort.sort import Sort
assert np.random.rand()==expected
tracker=Sort()
assert tracker.update(np.empty((0,5))).shape==(0,5)
import main
from tracking_manager import TrackingManager
assert TrackingManager().update([]).shape==(0,5)
from tools.evaluate_tracking import evaluate_tracking
from tools.benchmark_tracking import benchmark_tracking
assert not any(name.split('.')[0] in ('matplotlib','tkinter','skimage') for name in sys.modules)
print('headless import and tracking succeeded')
'''
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=str(ROOT),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('headless import', result.stdout)


class ApplicationTrackingTests(unittest.TestCase):
    def setUp(self):
        import test_stability as stability
        self.stability = stability
        self.app = stability.application
        fixture = stability.ApplicationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.prepare_stream = fixture.prepare_stream
        self.frame = fixture.frame

    def sequence(self, bottom_edges, age=1):
        self.prepare_stream([self.frame.copy() for _ in bottom_edges])
        self.app.mot_tracker = TrackingManager(TrackingConfig(age, 1, .3))
        self.app.traffic_light = 'red'
        self.app.coco_model.side_effect = [self.stability.result(
            [] if bottom is None else [[260, 20, 340, bottom, .9, 2]]) for bottom in bottom_edges]
        return self.app.generate_frames()

    def test_predicted_crossing_does_not_create_event_but_observed_recovery_can(self):
        stream = self.sequence([330, 348, None, 380])
        next(stream)
        next(stream)
        identity = next(iter(self.app.mot_tracker.tracks))
        self.assertEqual(self.app.crossing_detector.events, {})
        next(stream)
        state = self.app.mot_tracker.tracks[identity]
        self.assertFalse(state.observed)
        self.assertGreater(state.bbox[3] - 30, 320)
        self.assertEqual(self.app.crossing_detector.events, {})
        next(stream)
        self.assertIn(identity, self.app.crossing_detector.events)
        self.assertEqual(self.app.crossing_detector.events[identity]['frame_index'], 3)
        stream.close()

    def test_long_retention_does_not_override_phase2_observation_gap_limit(self):
        stream = self.sequence([330, 348, None, None, None, 400], age=5)
        for index in range(6):
            next(stream)
        self.assertEqual(self.app.mot_tracker.created_count, 1)
        self.assertEqual(self.app.crossing_detector.events, {})
        identity = next(iter(self.app.mot_tracker.tracks))
        self.assertEqual(self.app.crossing_detector.history[identity].status, 'uncertain')
        stream.close()

    def test_startup_custom_parameters_are_passed_to_sort_without_real_assets(self):
        fixture = self.stability.ApplicationTests()
        dependencies = fixture.dependencies()
        capture = self.stability.Capture([])
        with patch.dict(os.environ, {'TRAFFIC_SORT_MAX_AGE': '3', 'TRAFFIC_SORT_MIN_HITS': '2',
                                     'TRAFFIC_SORT_IOU_THRESHOLD': '.4'}), \
                patch.object(self.app, 'TRACKING_CONFIG', TrackingConfig()), \
                patch.object(self.app, '_require_file'), \
                patch.object(self.app.cv2, 'VideoCapture', return_value=capture), \
                patch.object(self.app, '_load_yolo_model', return_value=Mock()), \
                patch.dict(sys.modules, dependencies):
            self.app.initialize_resources()
            self.assertEqual(self.app.TRACKING_CONFIG, TrackingConfig(3, 2, .4))
        dependencies['sort.sort'].Sort.assert_called_once_with(max_age=3, min_hits=2, iou_threshold=.4)

    def test_expiration_and_shutdown_preserve_events_and_plate_information(self):
        stream = self.sequence([330, 348, 365, 380, None, None])
        for index in range(4):
            next(stream)
        identity, event = next(iter(self.app.crossing_detector.events.items()))
        event_id = event['event_id']
        self.app.crossing_detector.enrich_event(identity, 'AB12CDE', .9)
        self.app.plate_memory[identity] = 'AB12CDE'
        next(stream)
        next(stream)
        self.assertNotIn(identity, self.app.mot_tracker.tracks)
        list(stream)
        self.assertEqual(self.app.crossing_detector.events[identity]['event_id'], event_id)
        self.assertEqual(self.app.app.test_client().get('/violations').get_json(), ['AB12CDE'])
        self.assertEqual(self.app.mot_tracker.tracks, {})

    def test_new_identity_beyond_line_does_not_duplicate_old_candidate(self):
        stream = self.sequence([330, 348, 365, 380, None, 390, 400], age=0)
        for index in range(4):
            next(stream)
        old = next(iter(self.app.crossing_detector.events))
        next(stream)
        next(stream)
        next(stream)
        self.assertEqual(len(self.app.crossing_detector.events), 1)
        self.assertNotIn(old, self.app.mot_tracker.tracks)
        stream.close()

    def test_distinct_nearby_crossings_remain_separate_even_with_same_ocr_text(self):
        self.prepare_stream([self.frame.copy() for _ in range(4)])
        self.app.mot_tracker = TrackingManager(TrackingConfig(1, 1, .3))
        self.app.traffic_light = 'red'
        self.app.coco_model.side_effect = [self.stability.result(
            [[260, 20, 340, y, .9, 2], [350, 20, 430, y, .9, 2]]) for y in (330, 348, 365, 380)]
        self.assertEqual(len(list(self.app.generate_frames())), 4)
        events = self.app.crossing_detector.events
        self.assertEqual(len(events), 2)
        for identity in events:
            self.app.crossing_detector.enrich_event(identity, 'AB12CDE', .9)
        self.assertEqual(len({event['event_id'] for event in events.values()}), 2)

    def test_duplicate_outputs_clear_only_transient_crossing_history(self):
        self.prepare_stream(tracks=[[260, 20, 340, 370, 7], [360, 20, 440, 370, 7]])
        detector = self.app.crossing_detector
        detector.update(7, (300, 310), 0, 'red')
        detector.update(7, (300, 340), 1, 'red')
        event_id = detector.events[7]['event_id']
        self.app._frame_index = 1
        self.assertEqual(len(list(self.app.generate_frames())), 1)
        self.assertNotIn(7, detector.history)
        self.assertEqual(detector.events[7]['event_id'], event_id)
        self.app.license_plate_detector_model.assert_not_called()

    def test_invalid_tracking_configuration_returns_503_before_capture(self):
        with patch.dict(os.environ, {'TRAFFIC_SORT_MAX_AGE': '-1'}), \
                patch.object(self.app.cv2, 'VideoCapture') as capture:
            self.assertEqual(self.app.app.test_client().get('/video').status_code, 503)
        capture.assert_not_called()

    def test_configuration_is_frozen_for_an_active_session(self):
        self.prepare_stream()
        with patch.dict(os.environ, {'TRAFFIC_SORT_MIN_HITS': 'invalid'}):
            self.assertEqual(len(list(self.app.generate_frames())), 1)


if __name__ == '__main__':
    unittest.main()
