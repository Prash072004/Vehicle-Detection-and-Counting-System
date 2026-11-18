"""Offline evaluation correctness and benchmark input/resource safeguards."""

from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vehicle_detector import DetectionConfig
from tools import benchmark_detection as benchmark
from tools import detection_common as common
from tools import evaluate_detection as evaluation


def label(box=None, class_id=2):
    return {'class_id': class_id, 'bbox': [10, 10, 50, 50] if box is None else box}


def prediction(box=None, confidence=.9, class_id=2):
    return dict(label(box, class_id), confidence=confidence)


def dataset(boxes=None):
    return {'schema_version': 1, 'images': [{'id': 'a', 'width': 720, 'height': 480,
        'fully_annotated': True, 'boxes': [label()] if boxes is None else boxes}]}


def predictions(boxes=None):
    return {'schema_version': 1, 'model': {'name': 'synthetic-unit-test', 'version': 'fixture'},
            'images': [{'id': 'a', 'boxes': [prediction()] if boxes is None else boxes}]}


class EvaluationTests(unittest.TestCase):
    def evaluate(self, truth=None, detected=None, confidence=.25, iou=.5):
        return evaluation.evaluate_predictions(truth or dataset(), detected or predictions(),
                                                DetectionConfig(confidence=confidence), iou)

    def test_perfect_match(self):
        result = self.evaluate()['overall']
        self.assertEqual(result, {'true_positives': 1, 'false_positives': 0, 'false_negatives': 0,
                                   'precision': 1, 'recall': 1, 'f1': 1})

    def test_duplicate_prediction_is_false_positive(self):
        result = self.evaluate(detected=predictions([prediction(), prediction(confidence=.8)]))
        self.assertEqual(result['overall']['true_positives'], 1)
        self.assertEqual(result['overall']['false_positives'], 1)
        self.assertAlmostEqual(result['overall']['f1'], 2 / 3)

    def test_wrong_class_is_false_positive_and_false_negative(self):
        result = self.evaluate(detected=predictions([prediction(class_id=7)]))
        self.assertEqual(result['overall']['false_positives'], 1)
        self.assertEqual(result['overall']['false_negatives'], 1)
        self.assertEqual(result['per_class']['car']['false_negatives'], 1)
        self.assertEqual(result['per_class']['truck']['false_positives'], 1)

    def test_iou_matching_boundary(self):
        # Intersection 20*40, union 40*40 -> IoU exactly .5.
        detected = predictions([prediction([10, 10, 30, 50])])
        self.assertEqual(self.evaluate(detected=detected, iou=.5)['overall']['true_positives'], 1)
        self.assertEqual(self.evaluate(detected=detected, iou=.501)['overall']['true_positives'], 0)

    def test_confidence_boundary_and_missed_detection(self):
        detected = predictions([prediction(confidence=.25)])
        self.assertEqual(self.evaluate(detected=detected)['overall']['true_positives'], 1)
        result = self.evaluate(detected=detected, confidence=.251)
        self.assertEqual(result['overall']['false_negatives'], 1)
        self.assertIsNone(result['overall']['precision'])

    def test_empty_fully_labeled_image_can_measure_false_positives(self):
        result = self.evaluate(truth=dataset([]))['overall']
        self.assertEqual(result['false_positives'], 1)
        self.assertIsNone(result['recall'])

    def test_empty_predictions_are_not_fabricated_perfect_metrics(self):
        result = self.evaluate(truth=dataset([]), detected=predictions([]))['overall']
        self.assertIsNone(result['precision'])
        self.assertIsNone(result['recall'])
        self.assertIsNone(result['f1'])

    def test_partial_annotation_images_are_excluded(self):
        truth = dataset()
        truth['images'].append({'id': 'partial', 'width': 720, 'height': 480,
                                'fully_annotated': False, 'boxes': []})
        detected = predictions()
        detected['images'].append({'id': 'partial', 'boxes': [prediction()]})
        result = self.evaluate(truth, detected)
        self.assertEqual(result['evaluated_images'], 1)
        self.assertEqual(result['excluded_incomplete_images'], ['partial'])
        self.assertEqual(result['overall']['false_positives'], 0)

    def test_no_fully_annotated_images_prevents_accuracy_claim(self):
        truth = dataset()
        truth['images'][0]['fully_annotated'] = False
        with self.assertRaisesRegex(ValueError, 'cannot be calculated'):
            self.evaluate(truth)

    def test_missing_predictions_are_an_error_not_silent_false_negatives(self):
        detected = predictions()
        detected['images'] = []
        with self.assertRaisesRegex(ValueError, 'Missing predictions'):
            self.evaluate(detected=detected)

    def test_unknown_and_duplicate_prediction_ids_fail(self):
        for ids in (['unknown'], ['a', 'a']):
            detected = predictions()
            detected['images'] = [{'id': name, 'boxes': []} for name in ids]
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.evaluate(detected=detected)

    def test_ambiguous_annotations_are_rejected(self):
        for change in ({'fully_annotated': None}, {'width': 0}, {'boxes': [label([-1, 10, 50, 50])]},
                       {'boxes': [label(class_id=0)]}, {'boxes': [label([10, 10, float('nan'), 50])]}):
            truth = dataset()
            truth['images'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.evaluate(truth)

    def test_invalid_iou_thresholds_fail(self):
        for threshold in (0, -1, 1.1, float('nan')):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.evaluate(iou=threshold)

    def test_model_metadata_is_required(self):
        detected = predictions()
        del detected['model']
        with self.assertRaisesRegex(ValueError, 'model name and version'):
            self.evaluate(detected=detected)

    def test_malformed_json_structures_are_rejected(self):
        for value in (None, [], {'schema_version': 1, 'images': [None]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluation.validate_annotations(value)
        with self.assertRaises(ValueError):
            evaluation.evaluate_predictions(dataset(), [], DetectionConfig())

    def test_image_inference_keeps_annotation_coordinates_and_input_bytes(self):
        from test_stability import result
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            image = root / 'frame.png'
            cv2.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8))
            original = image.read_bytes()
            truth = dataset()
            truth['images'][0].update(width=200, height=100, file='frame.png')
            model = Mock(return_value=result([[10, 10, 50, 50, .9, 2]]))
            config = DetectionConfig(inference_size=960, device='cpu')
            output = evaluation.predict_images(truth, root / 'annotations.json', model, config)
            self.assertEqual(model.call_args[0][0].shape, (100, 200, 3))
            self.assertEqual(model.call_args[1]['imgsz'], 960)
            self.assertEqual(output[0]['boxes'][0]['bbox'], [10, 10, 50, 50])
            self.assertEqual(original, image.read_bytes())

    def test_missing_evaluation_image_fails_before_model_inference(self):
        truth = dataset()
        truth['images'][0]['file'] = '__missing_evaluation_image__.png'
        model = Mock()
        with self.assertRaisesRegex(ValueError, 'image is missing'):
            evaluation.predict_images(truth, ROOT / 'annotations.json', model, DetectionConfig())
        model.assert_not_called()

    def test_highest_score_gets_first_match_deterministically(self):
        truth = dataset([label([0, 0, 100, 100]), label([50, 0, 150, 100])])
        detected = predictions([prediction([0, 0, 100, 100], confidence=.8),
                                prediction([25, 0, 125, 100], confidence=.9)])
        original_truth, original_detected = deepcopy(truth), deepcopy(detected)
        result = self.evaluate(truth, detected)
        self.assertEqual(result['overall']['true_positives'], 1)
        self.assertEqual(result, self.evaluate(truth, detected))
        self.assertEqual(truth, original_truth)
        self.assertEqual(detected, original_detected)

    def test_invalid_predictions_use_same_boundary_as_application(self):
        detected = predictions([prediction(confidence=float('nan')), prediction([0, 0, 0, 10]),
                                prediction(class_id=0), prediction()])
        result = self.evaluate(detected=detected)
        self.assertEqual(result['overall']['true_positives'], 1)
        self.assertEqual(result['returned_prediction_filter_counts']['invalid'], 2)

    def test_json_csv_exports_include_metrics_and_model_configuration(self):
        result = self.evaluate()
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            evaluation.export_result(root / 'metrics.json', result)
            evaluation.export_result(root / 'metrics.csv', result)
            self.assertEqual(json.loads((root / 'metrics.json').read_text())['overall'], result['overall'])
            with (root / 'metrics.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertEqual(json.loads(rows[0]['configuration_json'])['model']['name'], 'synthetic-unit-test')
            with self.assertRaises(ValueError):
                evaluation.export_result(root / 'metrics.csv', result)

    def test_saved_prediction_cli_leaves_dataset_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            truth, detected = root / 'truth.json', root / 'predictions.json'
            truth.write_text(json.dumps(dataset()), encoding='utf-8')
            detected.write_text(json.dumps(predictions()), encoding='utf-8')
            original = truth.read_bytes(), detected.read_bytes()
            with patch('sys.stdout', new=io.StringIO()):
                code = evaluation.main(['--annotations', str(truth), '--predictions', str(detected),
                                        '--output', str(root / 'metrics.json')])
            self.assertEqual(code, 0)
            self.assertEqual(original, (truth.read_bytes(), detected.read_bytes()))


class BenchmarkTests(unittest.TestCase):
    def test_warmup_is_excluded_from_counts_and_frames_are_unchanged(self):
        rows = np.asarray([[10, 20, 100, 200, .9, 2], [10, 20, float('nan'), 200, .9, 2]])
        result = SimpleNamespace(boxes=SimpleNamespace(data=rows),
                                 speed={'preprocess': 1, 'inference': 2, 'postprocess': 3})
        model = Mock(return_value=[result])
        frame = np.zeros((480, 720, 3), dtype=np.uint8)
        with patch.object(benchmark, 'synchronize'), patch.object(benchmark, 'hardware', return_value={}):
            report = benchmark.benchmark_pass(model, [frame, frame], DetectionConfig(), warmup=1)
        self.assertEqual(model.call_count, 3)
        self.assertEqual(report['frames'], 2)
        self.assertEqual(report['counts']['accepted'], 2)
        self.assertEqual(report['counts']['invalid'], 2)
        self.assertFalse(frame.any())

    def test_sampling_uses_own_capture_and_releases_it(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True, np.zeros((12, 16, 3), dtype=np.uint8)) for _ in range(3)]
        with patch.object(benchmark.cv2, 'VideoCapture', return_value=capture):
            frames, indices = benchmark.sample_frames(Path('local-video.mp4'), 2, 2)
        self.assertEqual(indices, [0, 2])
        self.assertEqual(frames[0].shape, (480, 720, 3))
        self.assertEqual(capture.read.call_count, 3)
        capture.release.assert_called_once()

    def test_unopenable_or_empty_benchmark_source_fails_and_releases(self):
        for opened in (False, True):
            capture = Mock()
            capture.isOpened.return_value = opened
            capture.read.return_value = (False, None)
            with self.subTest(opened=opened), patch.object(benchmark.cv2, 'VideoCapture', return_value=capture):
                with self.assertRaises(ValueError):
                    benchmark.sample_frames(Path('local-video.mp4'), 2, 1)
            capture.release.assert_called_once()

    def test_latency_statistics_have_correct_units(self):
        values = benchmark.latency_summary([.01, .02, .03])
        self.assertAlmostEqual(values['mean_ms'], 20)
        self.assertAlmostEqual(values['median_ms'], 20)
        self.assertAlmostEqual(values['p95_ms'], 29)

    def test_missing_model_is_rejected_before_yolo_can_download(self):
        module = ModuleType('ultralytics')
        module.YOLO = Mock()
        with patch.dict(sys.modules, {'ultralytics': module}), self.assertRaisesRegex(ValueError, 'missing'):
            common.load_model(ROOT / 'models/__missing_phase3__.pt')
        module.YOLO.assert_not_called()

    def test_incompatible_local_model_is_rejected(self):
        module = ModuleType('ultralytics')
        module.YOLO = Mock(return_value=SimpleNamespace(task='detect', names={0: 'custom_vehicle'}))
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / 'model.pt'
            path.write_bytes(b'unit-test-placeholder')
            with patch.dict(sys.modules, {'ultralytics': module}), self.assertRaisesRegex(ValueError, 'compatible COCO'):
                common.load_model(path)

    def test_missing_video_rejected_without_capture(self):
        with patch.object(benchmark.cv2, 'VideoCapture') as capture, \
                patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit) as error:
            benchmark.main(['--models', 'models/yolov8n.pt', '--video', 'video/__missing_phase3__.mp4',
                            '--output', 'outputs/__unused__.json'])
        self.assertEqual(error.exception.code, 1)
        capture.assert_not_called()

    def test_offline_environment_is_temporary(self):
        previous = os.environ.get('YOLO_AUTOINSTALL')
        with common.offline_environment():
            self.assertEqual(os.environ['YOLO_AUTOINSTALL'], 'false')
            self.assertEqual(os.environ['YOLO_OFFLINE'], 'true')
            scratch = Path(os.environ['YOLO_CONFIG_DIR'])
            self.assertTrue(scratch.is_dir())
        self.assertEqual(os.environ.get('YOLO_AUTOINSTALL'), previous)
        self.assertFalse(scratch.exists())

    def test_report_writer_never_overwrites_existing_input(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / 'input.json'
            path.write_text('original', encoding='utf-8')
            with self.assertRaises(ValueError):
                common.write_json(path, {'changed': True}, protected=[path])
            self.assertEqual(path.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
