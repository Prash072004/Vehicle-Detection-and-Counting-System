"""Shared helpers for optional offline detection tools; no Flask dependencies."""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import cv2

from vehicle_detector import DetectionConfig, NMS_IOU, RegionOfInterest, VEHICLE_NAMES, project_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSING_SIZE = (720, 480)


def add_detection_arguments(parser, roi=False):
    parser.add_argument('--conf', type=float, default=0.25, help='Vehicle confidence cutoff (default: 0.25)')
    parser.add_argument('--imgsz', type=int, default=640, help='YOLO letterbox size, multiple of 32')
    parser.add_argument('--device', default='cpu', help='cpu, mps, or a GPU index (default: cpu)')
    if roi:
        parser.add_argument('--roi', type=float, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'))
        parser.add_argument('--roi-anchor', choices=['center', 'bottom_center'], default='bottom_center')


def detection_config(args):
    bounds = getattr(args, 'roi', None)
    roi = RegionOfInterest(enabled=bounds is not None, bounds=bounds,
                           anchor=getattr(args, 'roi_anchor', 'bottom_center'))
    return DetectionConfig(args.conf, args.imgsz, args.device, roi)


def require_file(value, description):
    path = project_path(value, PROJECT_ROOT)
    if not path.is_file():
        raise ValueError('{} is missing or is not a local file: {}'.format(description, path))
    return path


def require_model(value):
    path = require_file(value, 'YOLO model')
    if path.suffix.lower() != '.pt':
        raise ValueError('These tools accept existing local PyTorch .pt detection weights only')
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def offline_environment():
    """No autoinstall, online probes or persistent tool caches for local runs."""
    with tempfile.TemporaryDirectory(prefix='traffic-detection-') as scratch:
        settings = {'YOLO_AUTOINSTALL': 'false', 'YOLO_OFFLINE': 'true',
                    'YOLO_CONFIG_DIR': scratch, 'MPLCONFIGDIR': scratch}
        previous = {key: os.environ.get(key) for key in settings}
        os.environ.update(settings)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def load_model(path):
    # Existence must be checked before YOLO sees a filename: missing known model
    # names can otherwise trigger Ultralytics' automatic weight download.
    path = require_model(path)
    from ultralytics import YOLO
    model = YOLO(str(path))
    if model.task != 'detect' or any(model.names.get(key) != value for key, value in VEHICLE_NAMES.items()):
        raise ValueError('Vehicle model must be a compatible COCO detector with class IDs 2,3,5,7')
    return model


def synchronize(model):
    import torch
    device = next(model.model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def hardware(model=None):
    import torch
    import ultralytics
    import numpy as np
    device = str(next(model.model.parameters()).device) if model is not None else None
    gpu = None
    if device and device.startswith('cuda'):
        gpu = torch.cuda.get_device_name(next(model.model.parameters()).device)
    return {'platform': platform.platform(), 'cpu': platform.processor(),
            'logical_cpus': os.cpu_count(), 'torch_threads': torch.get_num_threads(),
            'python': platform.python_version(), 'ultralytics': ultralytics.__version__,
            'torch': torch.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__,
            'device': device, 'gpu': gpu}


def config_record(config):
    return dict(asdict(config), nms_iou=NMS_IOU)


def write_json(path, result, protected=()):
    path = project_path(path, PROJECT_ROOT)
    if path in {Path(p).resolve() for p in protected} or path.exists():
        raise ValueError('Refusing to overwrite an input or existing report: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
