"""Replay identical recorded detections through SORT parameter experiments."""

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy
import filterpy

from tracking_manager import TrackingConfig, TrackingManager, sanitize_tracks, validate_detections
from vehicle_detector import validate_probability
from tools.detection_common import require_file, sha256_file, write_json
from tools.evaluate_tracking import evaluate_tracking, validate_sequence


def parse_config(value):
    try:
        age, hits, threshold = value.split(',')
        return TrackingConfig(int(age), int(hits), float(threshold))
    except (ValueError, TypeError) as error:
        raise ValueError('Configuration must be max_age,min_hits,iou_threshold: ' + str(value)) from error


def _replay(frames, size, config, native):
    from sort.sort import Sort
    tracker = Sort(**asdict(config)) if native else TrackingManager(config, size)
    latencies, counts, records = [], [], []
    previous = set()
    created = expired = 0
    for index, detections in enumerate(frames):
        start = time.perf_counter()
        rows = tracker.update(detections)
        latencies.append(time.perf_counter() - start)
        if native:
            current = {track.id + 1 for track in tracker.trackers}
            rows, _ = sanitize_tracks(rows, size)
        else:
            current = set(tracker.tracks)
        created += len(current - previous)
        expired += len(previous - current)
        counts.append(len(current))
        previous = current
        records.append({'frame_index': index, 'objects': [
            {'id': int(row[4]), 'bbox': row[:4].tolist()} for row in rows]})
    return latencies, {'created': created, 'expired': expired,
        'active_mean': float(np.mean(counts)), 'active_peak': max(counts), 'active_final': counts[-1],
        'emitted_observations': sum(len(frame['objects']) for frame in records)}, records


def benchmark_tracking(data, config, repeats=3, warmup_replays=1, native=False, truth=None, iou=.5):
    size = validate_sequence(data, detections=True)
    validate_probability(iou, 'Evaluation IoU')
    if iou == 0:
        raise ValueError('Evaluation IoU must be greater than zero')
    if truth is not None:
        truth_size = validate_sequence(truth, ground_truth=True)
        if truth_size != size or len(truth['frames']) != len(data['frames']):
            raise ValueError('Annotations must cover exactly the same frames and size')
    if type(repeats) is not int or repeats < 1 or type(warmup_replays) is not int or warmup_replays < 0:
        raise ValueError('Repeats must be positive and warmup replays nonnegative integers')
    frames = [validate_detections(frame['detections'], size) for frame in data['frames']]
    for index in range(warmup_replays):
        _replay(frames, size, config, native)
    elapsed = []
    first_records = first_counts = None
    for repeat in range(repeats):
        timings, counts, records = _replay(frames, size, config, native)
        elapsed.extend(timings)
        if first_records is None:
            first_records, first_counts = records, counts
        elif counts != first_counts:
            raise RuntimeError('Nondeterministic replay counts; do not compare these runs')
    metadata = {'model': data.get('metadata', {}).get('model', 'unspecified'),
                'input_configuration': data.get('metadata', {}), 'tracker': asdict(config),
                'output_policy': 'native_SORT_with_startup_bypass' if native else 'confirmed_observed_only',
                'class_matching': 'Class agnostic: SORT does not retain detection class IDs'}
    predictions = {'schema_version': 1, 'frame_size': size, 'metadata': metadata, 'frames': first_records}
    return {'configuration': asdict(config), 'policy': metadata['output_policy'],
            'replays': repeats, 'warmup_replays': warmup_replays, 'frames_per_replay': len(frames),
            'latency_ms': {'mean': float(np.mean(elapsed) * 1000), 'median': float(np.median(elapsed) * 1000),
                           'p95': float(np.percentile(elapsed, 95) * 1000)},
            'tracker_only_fps': len(elapsed) / sum(elapsed), 'counts_per_replay': first_counts,
            'identity_metrics': evaluate_tracking(truth, predictions, iou, class_aware=False) if truth is not None else None,
            'accuracy_status': 'Measured on supplied identity annotations' if truth is not None else
                               'Unmeasured: no identity-labeled ground truth supplied',
            'predictions': predictions}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detections', required=True)
    parser.add_argument('--annotations', help='Optional complete identity labels; identical frame coverage')
    parser.add_argument('--configs', nargs='+', default=['1,3,0.3', '3,3,0.3', '5,3,0.3', '3,2,0.3'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup-replays', type=int, default=1)
    parser.add_argument('--compare-native', action='store_true')
    parser.add_argument('--include-tracks', action='store_true', help='Embed first-replay trajectories in the report')
    parser.add_argument('--iou', type=float, default=.5, help='Identity evaluation matching IoU')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        path = require_file(args.detections, 'Recorded detections')
        data = json.loads(path.read_text(encoding='utf-8'))
        validate_sequence(data, detections=True)
        configs = [parse_config(value) for value in args.configs]
        truth_path = require_file(args.annotations, 'Identity annotations') if args.annotations else None
        truth = json.loads(truth_path.read_text(encoding='utf-8')) if truth_path else None
        if truth is not None:
            validate_sequence(truth, ground_truth=True)
        report = {'schema_version': 1, 'detection_sha256': sha256_file(path),
            'input_metadata': data.get('metadata', {}),
            'annotation_sha256': sha256_file(truth_path) if truth_path else None,
            'annotation_metadata': truth.get('metadata', {}) if truth is not None else None,
            'hardware': {'platform': platform.platform(), 'cpu': platform.processor(),
                         'logical_cpus': os.cpu_count(), 'python': platform.python_version(),
                         'numpy': np.__version__, 'scipy': scipy.__version__,
                         'filterpy': filterpy.__version__,
                         'assignment_backend': 'lap' if importlib.util.find_spec('lap') else 'scipy'},
            'timing_scope': 'Timed tracker.update calls only. Adapter includes validation/lifecycle/output gates; '
                            'native mode includes SORT.update only. Excludes replay I/O, reporting, YOLO, '
                            'crossing, OCR, JPEG, HTTP and browser; not full application FPS.',
            'results': []}
        for config in configs:
            for native in ([True, False] if args.compare_native else [False]):
                result = benchmark_tracking(data, config, args.repeats, args.warmup_replays,
                                            native, truth, args.iou)
                if not args.include_tracks:
                    del result['predictions']
                report['results'].append(result)
                print('{} {} mean={:.3f}ms median={:.3f}ms p95={:.3f}ms tracker-only FPS={:.1f}'.format(
                    config, result['policy'], result['latency_ms']['mean'], result['latency_ms']['median'],
                    result['latency_ms']['p95'], result['tracker_only_fps']))
        write_json(args.output, report, protected=[path] + ([truth_path] if truth_path else []))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, 'Tracking benchmark failed: {}\n'.format(error))


if __name__ == '__main__':
    raise SystemExit(main())
