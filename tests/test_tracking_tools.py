"""Hand-calculated identity metric fixtures and fixed-detection replay tests."""

from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking_manager import TrackingConfig
from tools.evaluate_tracking import evaluate_tracking, export_report, main as evaluate_main, validate_sequence
from tools.benchmark_tracking import benchmark_tracking, main as benchmark_main, parse_config


def obj(identity, box=None, category=None):
    value = {'id': identity, 'bbox': [100, 100, 180, 200] if box is None else box}
    if category is not None:
        value['class_id'] = category
    return value


def sequence(objects, truth=False):
    return {'schema_version': 1, 'frame_size': [720, 480],
            'metadata': {'model': 'synthetic fixture; not field accuracy'},
            'frames': [dict(frame_index=index, objects=rows, **({'fully_annotated': True} if truth else {}))
                       for index, rows in enumerate(objects)]}


def detections():
    values = [[[100 + index, 100, 180 + index, 200, .9]] for index in range(6)] + [[], []]
    return {'schema_version': 1, 'frame_size': [720, 480], 'metadata': {'model': 'synthetic'},
            'frames': [{'frame_index': index, 'detections': rows} for index, rows in enumerate(values)]}


class IdentityMetricTests(unittest.TestCase):
    def test_perfect_track(self):
        truth = sequence([[obj(1)], [obj(1)], [obj(1)]], True)
        predicted = sequence([[obj(10)], [obj(10)], [obj(10)]])
        result = evaluate_tracking(truth, predicted)
        self.assertEqual((result['id_switches'], result['fragmentations']), (0, 0))
        self.assertEqual((result['identity_precision'], result['identity_recall'], result['idf1'], result['mota']), (1, 1, 1, 1))

    def test_identity_switch_is_not_fragmentation_when_continuously_visible(self):
        truth = sequence([[obj(1)], [obj(1)]], True)
        predicted = sequence([[obj(10)], [obj(20)]])
        result = evaluate_tracking(truth, predicted)
        self.assertEqual(result['id_switches'], 1)
        self.assertEqual(result['fragmentations'], 0)
        self.assertEqual(result['idf1'], .5)
        self.assertEqual(result['mota'], .5)

    def test_fragmentation_counts_recovery_after_missed_labeled_observation(self):
        truth = sequence([[obj(1)], [obj(1)], [obj(1)]], True)
        predicted = sequence([[obj(10)], [], [obj(10)]])
        result = evaluate_tracking(truth, predicted)
        self.assertEqual(result['fragmentations'], 1)
        self.assertEqual(result['id_switches'], 0)
        self.assertEqual(result['false_negatives'], 1)
        self.assertAlmostEqual(result['idf1'], .8)
        self.assertAlmostEqual(result['mean_continuity'], 2 / 3)

    def test_new_id_after_a_gap_counts_switch_and_fragmentation(self):
        truth = sequence([[obj(1)], [obj(1)], [obj(1)]], True)
        predicted = sequence([[obj(10)], [], [obj(20)]])
        result = evaluate_tracking(truth, predicted)
        self.assertEqual((result['id_switches'], result['fragmentations']), (1, 1))
        self.assertEqual(result['idf1'], .4)
        self.assertAlmostEqual(result['mota'], 1 / 3)

    def test_global_identity_assignment_is_not_greedy_or_local_matching(self):
        box_a, box_b = [100, 100, 180, 200], [300, 100, 380, 200]
        truth = sequence([[obj(1, box_a), obj(2, box_b)] for _ in range(5)], True)
        predicted = sequence([[obj(10, box_a)] for _ in range(3)] +
                             [[obj(10, box_b), obj(20, box_a)] for _ in range(2)])
        result = evaluate_tracking(truth, predicted)
        self.assertEqual(result['true_positives'], 7)
        self.assertEqual(result['id_true_positives'], 4)
        self.assertEqual(result['id_false_positives'], 3)
        self.assertEqual(result['id_false_negatives'], 6)
        self.assertAlmostEqual(result['idf1'], 8 / 17)

    def test_never_matched_track_counts_are_distinct_from_appearance_counts(self):
        truth = sequence([[obj(1), obj(2, [300, 100, 380, 200])] for _ in range(2)], True)
        predicted = sequence([[obj(10), obj(99, [500, 100, 580, 200])] for _ in range(2)])
        result = evaluate_tracking(truth, predicted)
        self.assertEqual((result['false_positive_tracks'], result['missed_tracks']), (1, 1))
        self.assertEqual((result['false_positives'], result['false_negatives']), (2, 2))

    def test_identity_switch_after_ground_truth_absence_is_not_fragmentation(self):
        truth = sequence([[obj(1)], [], [obj(1)]], True)
        predicted = sequence([[obj(10)], [], [obj(20)]])
        result = evaluate_tracking(truth, predicted)
        self.assertEqual(result['id_switches'], 1)
        self.assertEqual(result['fragmentations'], 0)

    def test_empty_predictions_count_misses_without_division_by_zero(self):
        result = evaluate_tracking(sequence([[obj(1)]], True), sequence([[]]))
        self.assertEqual(result['missed_tracks'], 1)
        self.assertEqual(result['mota'], 0)
        self.assertIsNone(result['identity_precision'])
        self.assertEqual(result['identity_recall'], 0)

    def test_empty_ground_truth_reports_false_positives_and_undefined_mota(self):
        result = evaluate_tracking(sequence([[]], True), sequence([[obj(10)]]))
        self.assertEqual(result['false_positive_tracks'], 1)
        self.assertIsNone(result['mota'])
        self.assertIsNone(result['identity_recall'])
        self.assertEqual(result['identity_precision'], 0)

    def test_both_empty_has_undefined_identity_scores(self):
        result = evaluate_tracking(sequence([[], []], True), sequence([[], []]))
        self.assertIsNone(result['idf1'])
        self.assertIsNone(result['mean_continuity'])
        self.assertEqual(result['id_switches'], 0)

    def test_iou_boundary_and_class_aware_matching(self):
        truth = sequence([[obj(1, [100, 100, 180, 200], 2)]], True)
        predicted = sequence([[obj(10, [100, 100, 140, 200], 7)]])
        self.assertEqual(evaluate_tracking(truth, predicted, .5)['true_positives'], 1)
        self.assertEqual(evaluate_tracking(truth, predicted, .501)['true_positives'], 0)
        self.assertEqual(evaluate_tracking(truth, predicted, .5, class_aware=True)['true_positives'], 0)

    def test_missing_classes_fail_when_class_aware_requested(self):
        with self.assertRaisesRegex(ValueError, 'class_id'):
            evaluate_tracking(sequence([[obj(1)]], True), sequence([[obj(10)]]), class_aware=True)

    def test_incomplete_labels_cannot_produce_identity_accuracy(self):
        truth = sequence([[obj(1)]], True)
        truth['frames'][0]['fully_annotated'] = False
        with self.assertRaisesRegex(ValueError, 'fully annotated'):
            evaluate_tracking(truth, sequence([[obj(10)]]))

    def test_frame_gaps_and_coverage_mismatch_fail(self):
        truth = sequence([[obj(1)], []], True)
        with self.assertRaisesRegex(ValueError, 'exactly the same frames'):
            evaluate_tracking(truth, sequence([[obj(10)]]))
        truth['frames'][1]['frame_index'] = 2
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            validate_sequence(truth, ground_truth=True)

    def test_duplicate_and_invalid_ids_and_boxes_fail(self):
        for objects in ([obj(1), obj(1)], [obj(0)], [obj(1.5)], [obj(1, [10, 10, float('nan'), 20])],
                        [obj(1, [-1, 10, 100, 100])], [obj(1, [10, 10, 10, 100])]):
            with self.subTest(objects=objects), self.assertRaises(ValueError):
                validate_sequence(sequence([objects]))

    def test_class_changes_on_a_ground_truth_identity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cannot change class'):
            validate_sequence(sequence([[obj(1, category=2)], [obj(1, category=7)]], True), ground_truth=True)

    def test_invalid_evaluation_thresholds_fail(self):
        for threshold in (0, -1, 2, float('nan')):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                evaluate_tracking(sequence([[]], True), sequence([[]]), threshold)

    def test_metrics_are_deterministic_and_do_not_mutate_inputs(self):
        truth = sequence([[obj(2), obj(1)]], True)
        predicted = sequence([[obj(20), obj(10)]])
        originals = deepcopy(truth), deepcopy(predicted)
        first = evaluate_tracking(truth, predicted)
        second = evaluate_tracking(truth, predicted)
        self.assertEqual(first, second)
        self.assertEqual(originals, (truth, predicted))

    def test_json_and_csv_cli_exports_preserve_inputs_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            truth, predicted = root / 'labels.json', root / 'tracks.json'
            truth.write_text(json.dumps(sequence([[obj(1)]], True)), encoding='utf-8')
            predicted.write_text(json.dumps(sequence([[obj(10)]])), encoding='utf-8')
            before = truth.read_bytes(), predicted.read_bytes()
            with patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(evaluate_main(['--annotations', str(truth), '--predictions', str(predicted),
                                               '--output', str(root / 'metrics.json')]), 0)
            report = json.loads((root / 'metrics.json').read_text())
            self.assertEqual(report['metrics']['idf1'], 1)
            export_report(root / 'metrics.csv', report, [truth, predicted])
            with (root / 'metrics.csv').open(newline='') as handle:
                self.assertEqual(next(csv.reader(handle)), ['field', 'value_json'])
            with self.assertRaises(ValueError):
                export_report(root / 'metrics.csv', report)
            self.assertEqual(before, (truth.read_bytes(), predicted.read_bytes()))


class ParameterBenchmarkTests(unittest.TestCase):
    def test_configuration_parser(self):
        self.assertEqual(parse_config('3,2,0.4'), TrackingConfig(3, 2, .4))
        for value in ('3,2', '3.5,2,.3', '-1,2,.3', '3,0,.3', '3,2,nan'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_config(value)

    def test_repeated_fixed_input_replays_keep_counts_and_input_content(self):
        data = detections()
        original = deepcopy(data)
        result = benchmark_tracking(data, TrackingConfig(), repeats=2, warmup_replays=1)
        self.assertEqual(result['counts_per_replay']['created'], 1)
        self.assertEqual(result['counts_per_replay']['expired'], 1)
        self.assertEqual(result['counts_per_replay']['active_final'], 0)
        self.assertIsNone(result['identity_metrics'])
        self.assertEqual(data, original)
        self.assertAlmostEqual(result['tracker_only_fps'], 1000 / result['latency_ms']['mean'])
        self.assertLessEqual(result['latency_ms']['median'], result['latency_ms']['p95'])

    def test_native_and_confirmed_output_policies_are_explicit(self):
        data = detections()
        native = benchmark_tracking(data, TrackingConfig(), repeats=1, warmup_replays=0, native=True)
        adapter = benchmark_tracking(data, TrackingConfig(), repeats=1, warmup_replays=0)
        self.assertEqual(native['counts_per_replay']['created'], adapter['counts_per_replay']['created'])
        self.assertGreater(native['counts_per_replay']['emitted_observations'],
                           adapter['counts_per_replay']['emitted_observations'])
        self.assertNotEqual(native['policy'], adapter['policy'])

    def test_labeled_synthetic_sequence_can_measure_metrics(self):
        data = detections()
        truth = sequence([[obj(1, frame['detections'][0][:4])] if frame['detections'] else []
                          for frame in data['frames']], True)
        result = benchmark_tracking(data, TrackingConfig(), repeats=1, warmup_replays=0, truth=truth)
        self.assertIsNotNone(result['identity_metrics']['idf1'])
        self.assertIn('supplied identity annotations', result['accuracy_status'])

    def test_invalid_replay_parameters_fail(self):
        for repeats, warmup in ((0, 0), (1, -1), (True, 0)):
            with self.subTest(repeats=repeats, warmup=warmup), self.assertRaises(ValueError):
                benchmark_tracking(detections(), TrackingConfig(), repeats, warmup)

    def test_invalid_or_mismatched_labels_fail_before_replay(self):
        for truth in ({}, sequence([[]], True)):
            with self.subTest(truth=truth), self.assertRaises(ValueError):
                benchmark_tracking(detections(), TrackingConfig(), truth=truth)
        with self.assertRaises(ValueError):
            benchmark_tracking(detections(), TrackingConfig(), iou=float('nan'))

    def test_benchmark_cli_exports_only_new_reports(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            source = root / 'detections.json'
            source.write_text(json.dumps(detections()), encoding='utf-8')
            original = source.read_bytes()
            output = root / 'benchmark.json'
            args = ['--detections', str(source), '--configs', '1,3,0.3', '--repeats', '1',
                    '--warmup-replays', '0', '--include-tracks', '--output', str(output)]
            with patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(benchmark_main(args), 0)
            self.assertIn('predictions', json.loads(output.read_text())['results'][0])
            with patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=io.StringIO()), \
                    self.assertRaises(SystemExit):
                benchmark_main(args)
            self.assertEqual(source.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
