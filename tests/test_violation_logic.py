"""Deterministic crossing trajectories and mocked HTTP/OCR integration."""
from fastapi.testclient import TestClient

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from violation_logic import CrossingConfig, CrossingDetector, crossing_anchor, segment_intersection


class CrossingLogicTests(unittest.TestCase):
    def setUp(self):
        self.detector = CrossingDetector()

    def cross(self, signal='red', track_id=1, x=300, first_frame=None, last_frame=None):
        if first_frame is None:
            first_frame = self.detector._last_frame_index + 1
        if last_frame is None:
            last_frame = first_frame + 1
        self.detector.update(track_id, (x, 310), first_frame, signal)
        return self.detector.update(track_id, (x, 330), last_frame, signal)

    def test_red_crossing_creates_one_candidate(self):
        result = self.cross()
        event = result['event']
        self.assertEqual(result['status'], 'candidate')
        self.assertEqual(event['crossing_point'], [300, 320])
        self.assertEqual(event['signal_state'], 'red')
        self.assertEqual(event['frame_index'], 1)
        self.assertEqual(event['status'], 'candidate')
        self.assertIsNone(event['license_plate'])
        self.assertNotEqual(event['event_id'], event['track_id'])
        self.assertEqual(len(self.detector.events), 1)

    def test_green_crossing_is_observed_without_candidate(self):
        self.assertEqual(self.cross('green')['status'], 'observed_crossing')
        self.assertEqual(self.detector.events, {})

    def test_first_sighting_already_beyond_line_is_uncertain(self):
        result = self.detector.update(1, (300, 350), 0, 'red')
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(self.detector.events, {})

    def test_stationary_before_or_after_line_never_creates_event(self):
        for y in (300, 350):
            with self.subTest(y=y):
                detector = CrossingDetector()
                for frame in range(5):
                    detector.update(1, (300, y), frame, 'red')
                self.assertEqual(detector.events, {})

    def test_reverse_direction_is_observed_but_not_a_candidate(self):
        self.detector.update(1, (300, 330), 0, 'red')
        result = self.detector.update(1, (300, 310), 1, 'red')
        self.assertEqual(result['status'], 'observed_crossing')
        self.assertEqual(self.detector.events, {})

    def test_prohibited_direction_is_configurable(self):
        detector = CrossingDetector(CrossingConfig(prohibited_direction='up'))
        detector.update(1, (300, 330), 0, 'red')
        self.assertEqual(detector.update(1, (300, 310), 1, 'red')['status'], 'candidate')

    def test_crossings_outside_segment_are_rejected(self):
        for x in (140, 560):
            self.cross(track_id=x, x=x)
        self.assertEqual(self.detector.events, {})

    def test_both_stop_line_endpoints_are_included(self):
        self.cross(track_id=1, x=160)
        self.cross(track_id=2, x=550)
        self.assertEqual(len(self.detector.events), 2)

    def test_endpoint_tolerance_is_small_and_configurable(self):
        self.cross(track_id=1, x=159)
        self.cross(track_id=2, x=157)
        self.assertEqual(set(self.detector.events), {1})
        strict = CrossingDetector(CrossingConfig(endpoint_tolerance_pixels=0))
        strict.update(1, (159, 310), 0, 'red')
        strict.update(1, (159, 330), 1, 'red')
        self.assertEqual(strict.events, {})

    def test_interpolated_x_is_used_instead_of_current_x(self):
        self.detector.update(1, (150, 300), 0, 'red')
        self.detector.update(1, (170, 340), 1, 'red')
        self.assertEqual(self.detector.events[1]['crossing_point'], [160, 320])
        other = CrossingDetector()
        other.update(1, (130, 300), 0, 'red')
        other.update(1, (170, 340), 1, 'red')
        self.assertEqual(other.events, {})

    def test_short_observation_gap_can_establish_crossing(self):
        self.assertEqual(self.cross(last_frame=3)['status'], 'candidate')

    def test_long_observation_gap_is_uncertain_without_retroactive_event(self):
        self.assertEqual(self.cross(last_frame=4)['status'], 'uncertain')
        self.detector.update(1, (300, 340), 5, 'red')
        self.assertEqual(self.detector.events, {})

    def test_large_unexplained_jump_is_uncertain(self):
        self.detector.update(1, (300, 200), 0, 'red')
        result = self.detector.update(1, (300, 400), 1, 'red')
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(self.detector.events, {})

    def test_multiple_vehicles_cross_independently(self):
        for track_id, x in ((1, 250), (2, 400)):
            self.detector.update(track_id, (x, 310), 0, 'red')
        for track_id, x in ((1, 250), (2, 400)):
            self.detector.update(track_id, (x, 330), 1, 'red')
        self.assertEqual(set(self.detector.events), {1, 2})
        self.assertEqual(len({event['event_id'] for event in self.detector.events.values()}), 2)

    def test_repeated_observations_and_recrossing_do_not_duplicate(self):
        event_id = self.cross()['event']['event_id']
        for frame, y in enumerate((340, 350, 310, 330, 340), 2):
            self.assertIsNone(self.detector.update(1, (300, y), frame, 'red')['event'])
        self.assertEqual(len(self.detector.events), 1)
        self.assertEqual(self.detector.events[1]['event_id'], event_id)

    def test_green_crossing_stays_legal_when_light_turns_red(self):
        self.cross('green')
        for frame in range(2, 5):
            self.detector.update(1, (300, 350), frame, 'red', signal_version=1)
        self.assertEqual(self.detector.events, {})

    def test_red_event_remains_when_light_turns_green(self):
        event = dict(self.cross()['event'])
        self.detector.update(1, (300, 350), 2, 'green', signal_version=1)
        self.assertEqual(self.detector.events[1], event)

    def test_stationary_beyond_line_when_signal_changes(self):
        self.detector.update(1, (300, 350), 0, 'green')
        self.detector.update(1, (300, 350), 1, 'red', signal_version=1)
        self.assertEqual(self.detector.events, {})

    def test_temporary_absence_within_gap_preserves_history(self):
        self.detector.update(1, (300, 310), 0, 'red')
        self.detector.expire(1)
        self.assertIn(1, self.detector.history)
        self.detector.update(1, (300, 330), 2, 'red')
        self.assertIn(1, self.detector.events)

    def test_expiration_does_not_remove_events_or_deduplication(self):
        self.cross()
        self.detector.expire(100)
        self.assertEqual(self.detector.history, {})
        self.assertIn(1, self.detector.events)
        self.detector.update(1, (300, 310), 100, 'red')
        self.detector.update(1, (300, 330), 101, 'red')
        self.assertEqual(len(self.detector.events), 1)

    def test_expired_unrecorded_history_cannot_establish_crossing(self):
        self.detector.update(1, (300, 310), 0, 'red')
        result = self.detector.update(1, (300, 330), 100, 'red')
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(self.detector.events, {})

    def test_jitter_inside_hysteresis_band_creates_no_event(self):
        self.detector.update(1, (300, 317), 0, 'red')
        for frame, y in enumerate([319.5, 320.5, 319.7, 320.8] * 5, 1):
            self.detector.update(1, (300, y), frame, 'red')
        self.assertEqual(self.detector.events, {})

    def test_signal_snapshot_at_intersection_survives_later_confirmation(self):
        for crossing_signal, later_signal in (('green', 'red'), ('red', 'green')):
            with self.subTest(signal=crossing_signal):
                detector = CrossingDetector()
                detector.update(1, (300, 317), 0, crossing_signal)
                detector.update(1, (300, 320.5), 1, crossing_signal)
                detector.update(1, (300, 323), 2, later_signal, signal_version=1)
                if crossing_signal == 'red':
                    self.assertEqual(detector.events[1]['signal_state'], 'red')
                    self.assertEqual(detector.events[1]['frame_index'], 1)
                else:
                    self.assertEqual(detector.events, {})

    def test_unknown_signal_never_becomes_red_retroactively(self):
        self.cross(signal='unknown')
        self.detector.update(1, (300, 340), 2, 'red', signal_version=1)
        self.assertEqual(self.detector.events, {})
        self.assertEqual(self.detector.recorded_crossings[1]['status'], 'uncertain')

    def test_signal_transition_during_observation_interval_is_uncertain(self):
        self.detector.update(1, (300, 310), 0, 'green')
        result = self.detector.update(1, (300, 330), 1, 'red', signal_version=1)
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(self.detector.events, {})

    def test_signal_version_detects_red_green_red_between_observations(self):
        self.detector.update(1, (300, 310), 0, 'red', signal_version=0)
        result = self.detector.update(1, (300, 330), 1, 'red', signal_version=2)
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(self.detector.events, {})

    def test_new_id_cannot_inherit_another_tracks_position(self):
        self.detector.update(1, (300, 310), 0, 'red')
        self.detector.update(2, (300, 330), 1, 'red')
        self.assertEqual(self.detector.events, {})

    def test_history_contains_previous_current_frames_and_last_seen(self):
        self.cross()
        state = self.detector.history[1]
        self.assertEqual(state.previous_position, (300, 310))
        self.assertEqual(state.current_position, (300, 330))
        self.assertEqual((state.previous_frame_index, state.current_frame_index, state.last_seen_frame), (0, 1, 1))
        self.assertTrue(state.crossing_recorded)

    def test_timestamp_is_interpolated_when_available(self):
        self.detector.update(1, (300, 310), 0, 'red', timestamp_ms=100)
        result = self.detector.update(1, (300, 330), 1, 'red', timestamp_ms=200)
        self.assertEqual(result['event']['timestamp_ms'], 150)

    def test_unknown_or_nonadvancing_timestamps_remain_unknown(self):
        self.detector.update(1, (300, 310), 0, 'red', timestamp_ms=0)
        self.detector.update(1, (300, 330), 1, 'red', timestamp_ms=0)
        self.assertIsNone(self.detector.events[1]['timestamp_ms'])

    def test_anchor_preserves_default_and_accepts_explicit_bottom_center(self):
        self.assertEqual(crossing_anchor((260, 100, 340, 350), CrossingConfig()), (300, 320))
        self.assertEqual(crossing_anchor((260, 100, 340, 350), CrossingConfig(anchor_offset=0)), (300, 350))

    def test_invalid_anchor_breaks_history_conservatively(self):
        self.detector.update(1, (300, 310), 0, 'red')
        result = self.detector.observe_bbox(1, (10, 10, float('nan'), 40), 1, 'red')
        self.assertEqual(result['status'], 'uncertain')
        self.assertIsNone(result['anchor'])
        self.assertNotIn(1, self.detector.history)
        self.detector.update(1, (300, 330), 2, 'red')
        self.assertEqual(self.detector.events, {})

    def test_sloped_and_vertical_stop_lines(self):
        config = CrossingConfig(line_start=(160, 300), line_end=(550, 340))
        detector = CrossingDetector(config)
        detector.update(1, (300, 300), 0, 'red')
        detector.update(1, (300, 340), 1, 'red')
        self.assertAlmostEqual(detector.events[1]['crossing_point'][1], 300 + 140 * 40 / 390)
        vertical = CrossingDetector(CrossingConfig(line_start=(320, 100), line_end=(320, 400),
                                                    prohibited_direction='right'))
        vertical.update(1, (310, 200), 0, 'red')
        vertical.update(1, (330, 200), 1, 'red')
        self.assertEqual(vertical.events[1]['crossing_point'], [320, 200])

    def test_reversing_line_endpoints_does_not_reverse_prohibited_direction(self):
        self.detector = CrossingDetector(CrossingConfig(line_start=(550, 320), line_end=(160, 320)))
        self.assertEqual(self.cross()['status'], 'candidate')

    def test_nonprohibited_sloped_crossing_cannot_be_confirmed_by_a_later_turn(self):
        detector = CrossingDetector(CrossingConfig(line_start=(160, 300), line_end=(550, 340)))
        detector.update(1, (300, 310), 0, 'red')
        detector.update(1, (330, 315), 1, 'red')
        # This segment crosses the sloped line while traveling upward.
        result = detector.update(1, (290, 314), 2, 'red')
        self.assertEqual(result['status'], 'observed_crossing')
        detector.update(1, (290, 320), 3, 'red')
        self.assertEqual(detector.events, {})

    def test_parallel_or_collinear_motion_does_not_cross(self):
        self.assertIsNone(segment_intersection((200, 320), (300, 320), (160, 320), (550, 320)))
        self.assertIsNone(segment_intersection((200, 300), (300, 300), (160, 320), (550, 320)))

    def test_exact_line_contact_with_hysteresis_disabled(self):
        self.detector = CrossingDetector(CrossingConfig(hysteresis_pixels=0))
        self.detector.update(1, (300, 310), 0, 'red')
        self.assertEqual(self.detector.update(1, (300, 320), 1, 'red')['status'], 'candidate')

    def test_minimum_movement_prevents_negligible_crossing(self):
        detector = CrossingDetector(CrossingConfig(hysteresis_pixels=0, minimum_movement_pixels=3))
        detector.update(1, (300, 319), 0, 'red')
        detector.update(1, (300, 320), 1, 'red')
        self.assertEqual(detector.events, {})

    def test_ocr_enrichment_preserves_candidate_and_best_information(self):
        self.cross()
        self.assertFalse(self.detector.enrich_event(1, None, None))
        self.assertIsNone(self.detector.events[1]['license_plate'])
        self.assertTrue(self.detector.enrich_event(1, 'AB12CDE', 0.9))
        self.assertFalse(self.detector.enrich_event(1, 'XX00XXX', 0.5))
        self.assertFalse(self.detector.enrich_event(1, '', 1))
        self.assertFalse(self.detector.enrich_event(1, 'XX00XXX', float('nan')))
        self.assertEqual(self.detector.events[1]['license_plate'], 'AB12CDE')
        self.assertEqual(self.detector.events[1]['status'], 'candidate')
        self.assertEqual(len(self.detector.events), 1)

    def test_configuration_validation(self):
        invalid = [dict(line_end=(160, 320)), dict(prohibited_direction='sideways'),
                   dict(line_start=(320, 100), line_end=(320, 400)),
                   dict(anchor_offset=-1), dict(hysteresis_pixels=-1),
                   dict(endpoint_tolerance_pixels=float('nan')), dict(maximum_frame_gap=0),
                   dict(maximum_frame_gap=True), dict(history_expiration_frames=1),
                   dict(maximum_jump_pixels=0), dict(frame_size=(0, 480)),
                   dict(line_start=(-1, 320))]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                CrossingConfig(**values)


class CrossingIntegrationTests(unittest.TestCase):
    def setUp(self):
        import test_stability as stability
        self.stability = stability
        self.application = stability.application
        fixture = stability.ApplicationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.prepare_stream = fixture.prepare_stream
        self.prepare_plate = fixture.prepare_plate
        self.frame = fixture.frame

    def test_first_sighting_beyond_line_does_not_run_ocr(self):
        app = self.application
        self.prepare_stream(tracks=[[260, 20, 340, 400, 7]])
        app.traffic_light = 'red'
        self.assertEqual(len(list(app.generate_frames())), 1)
        self.assertEqual(app.crossing_detector.events, {})
        app.license_plate_detector_model.assert_not_called()

    def test_ocr_no_result_preserves_event_and_legacy_json(self):
        app = self.application
        self.prepare_plate()
        app.read_license_plate.return_value = (None, None)
        list(app.generate_frames())
        self.assertEqual(app.crossing_detector.events[7]['status'], 'candidate')
        self.assertIsNone(app.crossing_detector.events[7]['license_plate'])
        self.assertEqual(TestClient(app.app, follow_redirects=False).get('/violations').json(), [])

    def test_ocr_success_enriches_same_event_and_legacy_json(self):
        app = self.application
        self.prepare_plate()
        list(app.generate_frames())
        event = app.crossing_detector.events[7]
        self.assertEqual(event['license_plate'], 'AB12CDE')
        self.assertEqual(event['license_plate_confidence'], 0.9)
        self.assertEqual(event['status'], 'candidate')
        self.assertEqual(TestClient(app.app, follow_redirects=False).get('/violations').json(), ['AB12CDE'])

    def test_ocr_exception_keeps_candidate_despite_terminal_stream_error(self):
        app = self.application
        self.prepare_plate()
        app.read_license_plate.side_effect = RuntimeError('OCR unavailable')
        self.assertEqual(list(app.generate_frames()), [])
        self.assertEqual(len(app.crossing_detector.events), 1)
        self.assertIsNone(app.crossing_detector.events[7]['license_plate'])

    def test_pending_ocr_can_succeed_after_light_turns_green(self):
        app = self.application
        capture = self.prepare_plate()
        capture.frames.append(self.frame.copy())
        capture.frame_count = 2
        app.read_license_plate.side_effect = [(None, None), ('AB12CDE', 0.9)]
        stream = app.generate_frames()
        next(stream)
        event_id = app.crossing_detector.events[7]['event_id']
        TestClient(app.app, follow_redirects=False).post('/set_light/green')
        next(stream)
        self.assertEqual(app.crossing_detector.events[7]['event_id'], event_id)
        self.assertEqual(app.crossing_detector.events[7]['signal_state'], 'red')
        self.assertEqual(app.crossing_detector.events[7]['license_plate'], 'AB12CDE')
        self.assertEqual(list(stream), [])

    def test_recognized_candidate_does_not_repeat_plate_inference(self):
        app = self.application
        capture = self.prepare_plate()
        capture.frames.extend([self.frame.copy(), self.frame.copy()])
        capture.frame_count = 3
        self.assertEqual(len(list(app.generate_frames())), 3)
        app.license_plate_detector_model.assert_called_once()
        app.read_license_plate.assert_called_once()

    def test_frame_signal_snapshot_is_shared_by_all_vehicles(self):
        app = self.application
        self.prepare_stream([self.frame.copy(), self.frame.copy()])
        app.traffic_light = 'red'
        app.mot_tracker.update.side_effect = [
            self.stability.np.asarray([[260, 20, 340, 340, 7], [360, 20, 440, 340, 8]]),
            self.stability.np.asarray([[260, 20, 340, 370, 7], [360, 20, 440, 370, 8]]),
        ]
        calls = []

        def infer(frame, **kwargs):
            calls.append(None)
            if len(calls) == 2:
                TestClient(app.app, follow_redirects=False).post('/set_light/green')
            return self.stability.result([])

        app.coco_model.side_effect = infer
        list(app.generate_frames())
        self.assertEqual(app.traffic_light, 'green')
        self.assertEqual(set(app.crossing_detector.events), {7, 8})
        self.assertTrue(all(event['signal_state'] == 'red' for event in app.crossing_detector.events.values()))

    def test_route_signal_versions_expose_intermediate_changes(self):
        app = self.application
        self.prepare_stream([self.frame.copy(), self.frame.copy()])
        app.traffic_light = 'red'
        app.mot_tracker.update.side_effect = [
            self.stability.np.asarray([[260, 20, 340, 340, 7]]),
            self.stability.np.asarray([[260, 20, 340, 370, 7]]),
        ]
        stream = app.generate_frames()
        next(stream)
        client = TestClient(app.app, follow_redirects=False)
        client.post('/set_light/green')
        client.post('/set_light/red')
        next(stream)
        self.assertEqual(app.crossing_detector.events, {})
        self.assertEqual(app.crossing_detector.recorded_crossings[7]['status'], 'uncertain')
        self.assertEqual(list(stream), [])

    def test_shared_frame_index_survives_browser_disconnect(self):
        app = self.application
        self.prepare_stream([self.frame.copy(), self.frame.copy()])
        app.traffic_light = 'red'
        app.mot_tracker.update.side_effect = [
            self.stability.np.asarray([[260, 20, 340, 340, 7]]),
            self.stability.np.asarray([[260, 20, 340, 370, 7]]),
        ]
        first = app.generate_frames()
        next(first)
        first.close()
        list(app.generate_frames())
        self.assertEqual(app.crossing_detector.events[7]['frame_index'], 1)

    def test_failed_jpeg_does_not_discard_crossing_event(self):
        app = self.application
        self.prepare_plate()
        with patch.object(app.cv2, 'imencode', return_value=(False, None)):
            self.assertEqual(list(app.generate_frames()), [])
        self.assertIn(7, app.crossing_detector.events)

    def test_invalid_anchor_is_skipped_without_crashing_stream(self):
        app = self.application
        self.prepare_stream(tracks=[[260, 20, float('nan'), 400, 7]])
        app.traffic_light = 'red'
        with patch.object(app.cvzone, 'cornerRect') as draw:
            self.assertEqual(len(list(app.generate_frames())), 1)
        draw.assert_not_called()
        self.assertEqual(app.crossing_detector.events, {})

    def test_nonadvancing_or_invalid_source_timestamps_are_unknown(self):
        app = self.application
        self.prepare_stream()
        app.cap.get = Mock(side_effect=[0, 0, 33, float('nan'), 66])
        self.assertEqual([app._video_timestamp_ms() for _ in range(5)], [0, None, 33, None, 66])


if __name__ == '__main__':
    unittest.main()
