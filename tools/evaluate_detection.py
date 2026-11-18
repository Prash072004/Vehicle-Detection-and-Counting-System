"""Deterministic class-aware IoU evaluation of explicitly complete annotations.

Dataset/prediction JSON schemas and limitations: docs/phase3_detection.md.
No Flask, SORT, OCR, training, dataset downloads, or dataset modifications.
"""

import argparse
import csv
import json
import math
from numbers import Real
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vehicle_detector import (DetectionCounts, VEHICLE_CLASSES, VEHICLE_NAMES,
                              prepare_detections, validate_frame_size, validate_probability)
from tools.detection_common import (PROJECT_ROOT, add_detection_arguments, config_record,
    detection_config, hardware, load_model, offline_environment, require_file, require_model,
    sha256_file, write_json)


def validate_annotations(dataset):
    if not isinstance(dataset, dict) or dataset.get('schema_version') != 1 or not isinstance(dataset.get('images'), list):
        raise ValueError('Expected schema_version 1 and an images list')
    ids = set()
    for image in dataset['images']:
        if not isinstance(image, dict):
            raise ValueError('Annotation images must be objects')
        image_id = image.get('id')
        if not isinstance(image_id, str) or not image_id or image_id in ids:
            raise ValueError('Image IDs must be unique nonempty strings')
        ids.add(image_id)
        width, height = validate_frame_size((image.get('width'), image.get('height')))
        if not isinstance(image.get('fully_annotated'), bool):
            raise ValueError('Every image must explicitly declare fully_annotated: true or false')
        if not isinstance(image.get('boxes'), list):
            raise ValueError('Every annotation image needs a boxes list')
        for label in image['boxes']:
            if not isinstance(label, dict):
                raise ValueError('Annotation boxes must be objects')
            class_id = label.get('class_id')
            if isinstance(class_id, bool) or class_id not in VEHICLE_CLASSES:
                raise ValueError('Labels must use COCO vehicle class IDs 2,3,5,7')
            box = label.get('bbox')
            if not isinstance(box, list) or len(box) != 4 or any(
                    isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in box):
                raise ValueError('Annotation bbox must contain four finite numeric coordinates')
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError('Annotation boxes must have positive area inside their image')
    if not any(image['fully_annotated'] for image in dataset['images']):
        raise ValueError('No fully annotated images; accuracy metrics cannot be calculated')


def intersection_over_union(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return overlap / (first_area + second_area - overlap)


def metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
            'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_predictions(dataset, predictions, config, iou_threshold=0.5):
    """Greedy score-ordered, one-to-one, same-class matching; ties use input order.

    Incomplete images are excluded entirely, including their predictions. Empty
    boxes on fully annotated images explicitly declare true negative frames.
    Undefined metric denominators are reported as null, never fabricated.
    """
    validate_annotations(dataset)
    validate_probability(iou_threshold, 'IoU threshold')
    if iou_threshold == 0:
        raise ValueError('IoU threshold must be greater than zero')
    if config.roi.enabled:
        raise ValueError('Evaluation uses the full labeled image; disable ROI')
    if not isinstance(predictions, dict) or predictions.get('schema_version') != 1 or not isinstance(predictions.get('images'), list):
        raise ValueError('Predictions need schema_version 1 and an images list')
    model = predictions.get('model')
    if not isinstance(model, dict) or any(not isinstance(model.get(key), str) or not model[key]
                                          for key in ('name', 'version')):
        raise ValueError('Predictions must record model name and version')
    known = {image['id'] for image in dataset['images']}
    by_id = {}
    for image in predictions['images']:
        if not isinstance(image, dict):
            raise ValueError('Prediction images must be objects')
        image_id = image.get('id')
        if not isinstance(image_id, str) or image_id not in known or image_id in by_id:
            raise ValueError('Prediction image IDs must be unique and present in annotations')
        if not isinstance(image.get('boxes'), list):
            raise ValueError('Prediction images need a boxes list')
        by_id[image_id] = image['boxes']
    totals = {class_id: [0, 0, 0] for class_id in VEHICLE_CLASSES}
    counts = DetectionCounts()
    evaluated = 0
    for image in dataset['images']:
        if not image['fully_annotated']:
            continue
        if image['id'] not in by_id:
            raise ValueError('Missing predictions (use an explicit empty list for no detections): ' + image['id'])
        evaluated += 1
        candidates = []
        for detection in by_id[image['id']]:
            if not isinstance(detection, dict) or not isinstance(detection.get('bbox'), list):
                raise ValueError('Prediction entries require bbox, confidence and class_id')
            row = detection['bbox'] + [detection.get('confidence'), detection.get('class_id')]
            valid = prepare_detections([row], (image['width'], image['height']), config, counts)
            if len(valid):
                candidates.append((valid[0].tolist(), int(detection['class_id'])))
        # Python's stable sort preserves annotation/prediction file order on ties.
        candidates.sort(key=lambda value: -value[0][4])
        matched = set()
        labels = image['boxes']
        for box, class_id in candidates:
            options = [(intersection_over_union(box, label['bbox']), index)
                       for index, label in enumerate(labels)
                       if index not in matched and label['class_id'] == class_id]
            options.sort(key=lambda value: (-value[0], value[1]))
            if options and options[0][0] >= iou_threshold:
                matched.add(options[0][1])
                totals[class_id][0] += 1
            else:
                totals[class_id][1] += 1
        for index, label in enumerate(labels):
            if index not in matched:
                totals[label['class_id']][2] += 1
    overall = [sum(value[index] for value in totals.values()) for index in range(3)]
    return {'schema_version': 1, 'model': model, 'configuration': config_record(config),
            'iou_threshold': iou_threshold, 'matching': 'Greedy descending confidence, same class, '
                'one-to-one, IoU >= threshold; stable input-order ties',
            'evaluated_images': evaluated,
            'excluded_incomplete_images': [image['id'] for image in dataset['images']
                                           if not image['fully_annotated']],
            'overall': metrics(*overall),
            'per_class': {VEHICLE_NAMES[key]: dict(class_id=key, **metrics(*value))
                          for key, value in totals.items()},
            'returned_prediction_filter_counts': vars(counts),
            'limitations': 'Single confidence/IoU operating point; no mAP, tracking, crossing, '
                           'OCR or legal-violation accuracy. Null means an undefined denominator.'}


def predict_images(dataset, annotation_path, model, config):
    import cv2
    results = []
    for image in dataset['images']:
        if not image['fully_annotated']:
            continue
        filename = image.get('file')
        if not isinstance(filename, str) or not filename or '://' in filename:
            raise ValueError('Real inference requires a local file for each evaluated image')
        path = Path(filename)
        path = path if path.is_absolute() else annotation_path.parent / path
        if not path.is_file():
            raise ValueError('Evaluation image is missing: ' + str(path))
        frame = cv2.imread(str(path))
        if frame is None or (frame.shape[1], frame.shape[0]) != (image['width'], image['height']):
            raise ValueError('Unreadable image or annotation/image size mismatch: ' + str(path))
        result = model(frame, **dict(config.inference_arguments(), verbose=False))[0]
        results.append({'id': image['id'], 'boxes': [
            {'bbox': row[:4], 'confidence': row[4], 'class_id': row[5]}
            for row in result.boxes.data.tolist()]})
    return results


def export_result(path, result, protected=()):
    path = Path(path)
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if path.suffix.lower() == '.json':
        write_json(path, result, protected)
    elif path.suffix.lower() == '.csv':
        if path.exists() or path in {Path(p).resolve() for p in protected}:
            raise ValueError('Refusing to overwrite an input or existing report: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [('overall', result['overall']), *result['per_class'].items()]
        with path.open('x', encoding='utf-8', newline='') as handle:
            fields = ['class', 'class_id', 'true_positives', 'false_positives', 'false_negatives',
                      'precision', 'recall', 'f1', 'configuration_json']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            metadata = {key: value for key, value in result.items() if key not in ('overall', 'per_class')}
            for name, values in rows:
                writer.writerow(dict(values, **{'class': name,
                    'configuration_json': json.dumps(metadata, sort_keys=True, allow_nan=False)}))
    else:
        raise ValueError('Evaluation output must have a .json or .csv extension')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', help='Existing local compatible COCO .pt weights')
    source.add_argument('--predictions', help='Saved predictions JSON; no inference required')
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--output', required=True, help='New .json or .csv report')
    parser.add_argument('--save-predictions', help='Optional new predictions JSON (with --model)')
    add_detection_arguments(parser)
    args = parser.parse_args(argv)
    try:
        config = detection_config(args)
        validate_probability(args.iou, 'IoU threshold')
        if args.iou == 0:
            raise ValueError('IoU threshold must be greater than zero')
        annotation_path = require_file(args.annotations, 'Annotations')
        dataset = json.loads(annotation_path.read_text(encoding='utf-8'))
        validate_annotations(dataset)
        protected = [annotation_path]
        if args.predictions:
            if args.save_predictions:
                raise ValueError('--save-predictions requires --model')
            path = require_file(args.predictions, 'Predictions')
            predictions = json.loads(path.read_text(encoding='utf-8'))
            protected.append(path)
        else:
            path = require_model(args.model)
            protected.append(path)
            with offline_environment():
                model = load_model(path)
                images = predict_images(dataset, annotation_path, model, config)
                predictions = {'schema_version': 1, 'model': {'name': path.name,
                    'sha256': sha256_file(path), 'version': hardware(model)['ultralytics']},
                    'configuration': config_record(config), 'hardware': hardware(model), 'images': images}
        report = evaluate_predictions(dataset, predictions, config, args.iou)
        report['annotation_sha256'] = sha256_file(annotation_path)
        report['prediction_source'] = {'type': 'saved_predictions' if args.predictions else 'local_model',
                                      'sha256': sha256_file(path)}
        report['inference_provenance'] = {key: predictions.get(key) for key in ('configuration', 'hardware')}
        export_result(args.output, report, protected)
        if args.save_predictions:
            write_json(args.save_predictions, predictions, protected=[*protected, Path(args.output).resolve()])
        print(json.dumps(report['overall'], indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, 'Evaluation failed: {}\n'.format(error))


if __name__ == '__main__':
    raise SystemExit(main())
