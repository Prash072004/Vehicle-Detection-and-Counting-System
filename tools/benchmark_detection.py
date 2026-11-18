"""Optional local model timing: python -B tools/benchmark_detection.py --help."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from vehicle_detector import DetectionCounts, VEHICLE_CLASSES, prepare_detections
from tools.detection_common import (PROCESSING_SIZE, add_detection_arguments, config_record,
    detection_config, hardware, load_model, offline_environment, require_file, require_model,
    sha256_file, synchronize, write_json)


def sample_frames(path, count, stride):
    """Own capture, bounded sample; no application capture, tracker, or OCR."""
    if count < 1 or stride < 1:
        raise ValueError('Frame count and stride must be positive')
    capture = cv2.VideoCapture(str(path))
    frames, indices = [], []
    try:
        if not capture.isOpened():
            raise ValueError('Could not open benchmark video: ' + str(path))
        for index in range(count * stride):
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or not frame.size:
                raise ValueError('Empty benchmark frame at index ' + str(index))
            if index % stride == 0:
                frames.append(cv2.resize(frame, PROCESSING_SIZE))
                indices.append(index)
                if len(frames) == count:
                    break
    finally:
        capture.release()
    if not frames:
        raise ValueError('No decodable benchmark frames')
    return frames, indices


def latency_summary(seconds):
    if not seconds:
        raise ValueError('At least one timing sample is required')
    values = np.asarray(seconds, dtype=float)
    return {'mean_ms': float(values.mean() * 1000),
            'median_ms': float(np.median(values) * 1000),
            'p95_ms': float(np.percentile(values, 95) * 1000)}


def benchmark_pass(model, frames, config, warmup, legacy=False):
    arguments = dict(config.inference_arguments(), verbose=False)
    for index in range(warmup):
        model(frames[index % len(frames)], **arguments)
    synchronize(model)
    latencies, stages = [], []
    counts = DetectionCounts()
    total_start = time.perf_counter()
    for frame in frames:
        synchronize(model)
        start = time.perf_counter()
        result = model(frame, **arguments)[0]
        synchronize(model)
        latencies.append(time.perf_counter() - start)
        stages.append(result.speed)
        rows = result.boxes.data.tolist()
        if legacy:
            # The pre-Phase-3 YOLO-to-SORT adapter; same inference arguments,
            # source frames and warmup as the updated adapter for fair timing.
            accepted = [list(row[:5]) for row in rows if int(row[5]) in VEHICLE_CLASSES]
            np.asarray(accepted, dtype=float).reshape(-1, 5)
            counts.returned_by_yolo += len(rows)
            counts.accepted += len(accepted)
            counts.unsupported_class += len(rows) - len(accepted)
        else:
            prepare_detections(rows, PROCESSING_SIZE, config, counts)
    elapsed = time.perf_counter() - total_start
    return {'pipeline': 'original_vehicle_adapter' if legacy else 'validated_vehicle_adapter',
            'frames': len(frames), 'warmup_frames': warmup,
            'inference_latency': latency_summary(latencies),
            'model_call_fps': len(frames) / sum(latencies),
            'detection_pipeline_fps': len(frames) / elapsed,
            'mean_yolo_stages_ms': {key: float(np.mean([stage[key] for stage in stages]))
                                    for key in ('preprocess', 'inference', 'postprocess')},
            'counts': asdict(counts), 'hardware': hardware(model)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True, help='Existing local COCO .pt weights only')
    parser.add_argument('--video', default='video/sample.mp4')
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--compare-baseline', action='store_true')
    parser.add_argument('--output', required=True, help='New JSON report path (will not overwrite)')
    add_detection_arguments(parser, roi=True)
    args = parser.parse_args(argv)
    try:
        config = detection_config(args)
        if args.frames < 1 or args.stride < 1 or args.warmup < 0:
            raise ValueError('Frames/stride must be positive and warmup nonnegative')
        if args.compare_baseline and config.roi.enabled:
            raise ValueError('Disable ROI for an equivalent baseline comparison')
        paths = [require_model(value) for value in args.models]
        video = require_file(args.video, 'Benchmark video')
        frames, indices = sample_frames(video, args.frames, args.stride)
        report = {'schema_version': 1, 'configuration': config_record(config),
                  'video': str(video), 'video_sha256': sha256_file(video),
                  'processing_size': PROCESSING_SIZE, 'sample_frame_indices': indices,
                  'timing_scope': 'Model call and detection adapter only; excludes video decoding, '
                                  'resizing, SORT, crossing logic, plate inference, OCR, JPEG and Flask. '
                                  'Neither FPS figure is complete application FPS.',
                  'counter_scope': 'Returned YOLO rows only; internal pre-NMS confidence removals '
                                   'are not observable. Original adapter does not validate boxes.',
                  'models': []}
        with offline_environment():
            for path in paths:
                start = time.perf_counter()
                model = load_model(path)
                initialization = time.perf_counter() - start
                passes = []
                if args.compare_baseline:
                    passes.append(benchmark_pass(model, frames, config, args.warmup, legacy=True))
                passes.append(benchmark_pass(model, frames, config, args.warmup))
                report['models'].append({'name': path.name, 'sha256': sha256_file(path),
                    'initialization_seconds': initialization,
                    'initialization_scope': 'Includes first-call Python library imports; warmup excluded',
                    'passes': passes})
                del model
        write_json(args.output, report, protected=[*paths, video])
        for entry in report['models']:
            for result in entry['passes']:
                print('{} {}: mean={:.2f} ms, median={:.2f} ms, p95={:.2f} ms, '
                      'detection-only FPS={:.2f}, detections={}'.format(entry['name'], result['pipeline'],
                      result['inference_latency']['mean_ms'], result['inference_latency']['median_ms'],
                      result['inference_latency']['p95_ms'], result['detection_pipeline_fps'],
                      result['counts']['accepted']))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, 'Benchmark failed: {}\n'.format(error))


if __name__ == '__main__':
    raise SystemExit(main())
