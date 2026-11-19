"""Offline identity evaluation of a complete, consecutively labeled sequence.

Format and metric conventions: docs/phase4_tracking.md. Uses installed SciPy;
does not import Flask, YOLO, OCR, or a plotting package. No dataset modifications.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import linear_sum_assignment

from vehicle_detector import validate_frame_size, validate_probability, VEHICLE_CLASSES
from tools.detection_common import PROJECT_ROOT, require_file, sha256_file, write_json


def validate_sequence(data, ground_truth=False, detections=False):
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('Tracking data needs schema_version 1')
    if not isinstance(data.get('metadata', {}), dict):
        raise ValueError('Tracking metadata must be an object')
    size = validate_frame_size(data.get('frame_size'))
    frames = data.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('A nonempty frames list is required')
    classes = {}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or type(frame.get('frame_index')) is not int or frame['frame_index'] != index:
            raise ValueError('Frames must be consecutive, ordered, and start at 0; include empty frames')
        if ground_truth and frame.get('fully_annotated') is not True:
            raise ValueError('Identity metrics require a fully annotated contiguous sequence')
        if detections:
            from tracking_manager import validate_detections
            if not isinstance(frame.get('detections'), list):
                raise ValueError('Each frame needs an explicit detections list')
            validate_detections(frame['detections'], size)
            continue
        objects = frame.get('objects')
        if not isinstance(objects, list):
            raise ValueError('Each frame needs an explicit objects list')
        ids = set()
        for obj in objects:
            if not isinstance(obj, dict):
                raise ValueError('Tracking objects must be dictionaries')
            identity = obj.get('id')
            if type(identity) is not int or identity <= 0 or identity in ids:
                raise ValueError('IDs must be positive integers, unique within each frame')
            ids.add(identity)
            box = obj.get('bbox')
            if not isinstance(box, list) or len(box) != 4 or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in box):
                raise ValueError('Bounding boxes need four finite numeric coordinates')
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
                raise ValueError('Boxes must have positive area inside the declared frame')
            category = obj.get('class_id')
            if category is not None:
                if type(category) is not int or category not in VEHICLE_CLASSES:
                    raise ValueError('class_id must be a supported COCO vehicle ID')
                if ground_truth and identity in classes and classes[identity] != category:
                    raise ValueError('A ground-truth identity cannot change class')
                classes[identity] = category
    return size


def _similarity(truth, predictions, class_aware):
    matrix = np.zeros((len(truth), len(predictions)))
    for row, first in enumerate(truth):
        for column, second in enumerate(predictions):
            if class_aware and first['class_id'] != second['class_id']:
                continue
            a, b = first['bbox'], second['bbox']
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
            matrix[row, column] = overlap / union
    return matrix


def evaluate_tracking(truth, predictions, iou_threshold=.5, class_aware=False):
    """CLEAR-style local matches plus global one-to-one identity assignment.

    Never estimate identities from unlabeled footage. IDF1 uses all eligible
    spatial pair counts across time, NOT local frame matches or greedy ID votes.
    """
    size = validate_sequence(truth, ground_truth=True)
    if validate_sequence(predictions) != size or len(truth['frames']) != len(predictions['frames']):
        raise ValueError('Ground truth and predictions must cover exactly the same frames and size')
    threshold = validate_probability(iou_threshold, 'Evaluation IoU')
    if threshold == 0:
        raise ValueError('Evaluation IoU must be greater than zero')
    if class_aware and any('class_id' not in obj or obj['class_id'] is None
            for data in (truth, predictions) for frame in data['frames'] for obj in frame['objects']):
        raise ValueError('Class-aware evaluation requires class_id on every object')
    gt_count, pr_count, pair_counts = Counter(), Counter(), Counter()
    matched_gt, matched_pr = Counter(), set()
    previous_frame, last_match = {}, {}
    lost_since_match = set()
    fragmentations = Counter()
    tp = fp = fn = switches = 0
    for gt_frame, pr_frame in zip(truth['frames'], predictions['frames']):
        gt = sorted(gt_frame['objects'], key=lambda obj: obj['id'])
        pr = sorted(pr_frame['objects'], key=lambda obj: obj['id'])
        gt_count.update(obj['id'] for obj in gt)
        pr_count.update(obj['id'] for obj in pr)
        similarities = _similarity(gt, pr, class_aware)
        valid = similarities >= threshold
        for row, column in zip(*np.nonzero(valid)):
            pair_counts[gt[row]['id'], pr[column]['id']] += 1
        matches = []
        if gt and pr:
            score = similarities.copy()
            # Preserve eligible immediately preceding frame matches first;
            # then maximize IoU. Sorted IDs make tied inputs reproducible.
            bonus = max(len(gt), len(pr)) + 1
            for row, obj in enumerate(gt):
                for column, candidate in enumerate(pr):
                    if previous_frame.get(obj['id']) == candidate['id']:
                        score[row, column] += bonus
            score[~valid] = 0
            rows, columns = linear_sum_assignment(-score)
            matches = [(int(row), int(column)) for row, column in zip(rows, columns) if valid[row, column]]
        current = {}
        for row, column in matches:
            gid, pid = gt[row]['id'], pr[column]['id']
            if gid in last_match and last_match[gid] != pid:
                switches += 1
            if gid in lost_since_match:
                fragmentations[gid] += 1
                lost_since_match.remove(gid)
            last_match[gid] = pid
            current[gid] = pid
            matched_gt[gid] += 1
            matched_pr.add(pid)
        for obj in gt:
            gid = obj['id']
            if gid in last_match and gid not in current:
                lost_since_match.add(gid)
        previous_frame = current
        tp += len(matches)
        fn += len(gt) - len(matches)
        fp += len(pr) - len(matches)
    gt_ids, pr_ids = sorted(gt_count), sorted(pr_count)
    identity_tp = 0
    if gt_ids and pr_ids:
        # Maximizing pair counts is equivalent to minimizing global IDFP+IDFN:
        # those equal total predicted + total GT appearances minus 2*IDTP.
        weights = np.array([[pair_counts[gid, pid] for pid in pr_ids] for gid in gt_ids], dtype=np.int64)
        rows, columns = linear_sum_assignment(-weights)
        identity_tp = int(weights[rows, columns].sum())
    total_gt, total_pr = sum(gt_count.values()), sum(pr_count.values())
    id_fn, id_fp = total_gt - identity_tp, total_pr - identity_tp
    identity = {'id_true_positives': identity_tp, 'id_false_positives': id_fp, 'id_false_negatives': id_fn,
                'identity_precision': identity_tp / total_pr if total_pr else None,
                'identity_recall': identity_tp / total_gt if total_gt else None,
                'idf1': 2 * identity_tp / (total_gt + total_pr) if total_gt + total_pr else None}
    continuity = {str(gid): {'labeled_appearances': gt_count[gid], 'matched_appearances': matched_gt[gid],
                    'matched_fraction': matched_gt[gid] / gt_count[gid],
                    'fragmentations': fragmentations[gid]} for gid in gt_ids}
    return {'frames': len(truth['frames']), 'ground_truth_appearances': total_gt,
            'predicted_appearances': total_pr, 'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
            'id_switches': switches, 'fragmentations': sum(fragmentations.values()),
            'false_positive_tracks': len(set(pr_ids) - matched_pr),
            'missed_tracks': len(set(gt_ids) - set(matched_gt)),
            'mota': 1 - (fn + fp + switches) / total_gt if total_gt else None,
            **identity, 'continuity': continuity,
            'mean_continuity': float(np.mean([value['matched_fraction'] for value in continuity.values()]))
                               if continuity else None,
            'configuration': {'iou_threshold': threshold, 'class_aware': class_aware,
                'matching': 'Previous-frame continuity priority, then Hungarian IoU; sorted ID ties',
                'id_switch': 'Different predicted ID from last matched observation, including across gaps',
                'fragmentation': 'Matched -> missed labeled appearance(s) -> matched; GT absence is not a miss',
                'identity': 'Global maximum spatially eligible pair counts with one-to-one identity assignment',
                'undefined': 'Null when denominator is zero; no official MOTChallenge preprocessing'}}


def export_report(path, result, protected=()):
    path = Path(path)
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if path.suffix.lower() == '.json':
        write_json(path, result, protected)
    elif path.suffix.lower() == '.csv':
        if path.exists() or path in {Path(p).resolve() for p in protected}:
            raise ValueError('Refusing to overwrite an existing report or input')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['field', 'value_json'])
            for key, value in result.items():
                writer.writerow([key, json.dumps(value, sort_keys=True, allow_nan=False)])
    else:
        raise ValueError('Report extension must be .json or .csv')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--class-aware', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        truth_path, prediction_path = require_file(args.annotations, 'Identity annotations'), require_file(args.predictions, 'Tracks')
        truth = json.loads(truth_path.read_text(encoding='utf-8'))
        predictions = json.loads(prediction_path.read_text(encoding='utf-8'))
        metrics = evaluate_tracking(truth, predictions, args.iou, args.class_aware)
        report = {'schema_version': 1, 'metrics': metrics,
                  'annotation_sha256': sha256_file(truth_path), 'prediction_sha256': sha256_file(prediction_path),
                  'annotation_metadata': truth.get('metadata', {}), 'prediction_metadata': predictions.get('metadata', {})}
        export_report(args.output, report, [truth_path, prediction_path])
        print(json.dumps({key: metrics[key] for key in ('id_switches', 'fragmentations', 'idf1', 'mota')}, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, 'Tracking evaluation failed: {}\n'.format(error))


if __name__ == '__main__':
    raise SystemExit(main())
