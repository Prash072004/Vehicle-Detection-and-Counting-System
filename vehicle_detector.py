"""Vehicle detection configuration and the validated YOLO-to-SORT boundary.

No models, video captures, or environment settings are loaded on import.
Coordinates always describe the input image, not YOLO's letterboxed tensor.
"""

from dataclasses import dataclass, replace
import math
from numbers import Real
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


VEHICLE_CLASSES = (2, 3, 5, 7)
VEHICLE_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
# Preserve the installed YOLOv8 default and make experiment provenance explicit.
NMS_IOU = 0.7


def _number(value, description):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(description + ' must be numeric')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(description + ' must be finite')
    return number


def validate_probability(value, description='Confidence'):
    value = _number(value, description)
    if not 0 <= value <= 1:
        raise ValueError(description + ' must be between 0 and 1')
    return value


def validate_frame_size(size):
    if not isinstance(size, (tuple, list)) or len(size) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in size):
        raise ValueError('Frame size must be (width, height) in positive integers')
    return tuple(size)


@dataclass(frozen=True)
class RegionOfInterest:
    """Inclusive rectangle membership by center or bottom-center of a clipped box.

    Bounds and reference_size use processing-image pixels. A partial overlap is
    accepted when its representative point is inside, including the boundary.
    This is independent of the crossing detector's anchor and stop line.
    """

    enabled: bool = False
    bounds: Optional[Tuple[float, float, float, float]] = None
    anchor: str = 'bottom_center'
    reference_size: Tuple[int, int] = (720, 480)

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('ROI enabled must be boolean')
        width, height = validate_frame_size(self.reference_size)
        object.__setattr__(self, 'reference_size', (width, height))
        if self.anchor not in ('center', 'bottom_center'):
            raise ValueError('ROI anchor must be center or bottom_center')
        if self.bounds is None:
            if self.enabled:
                raise ValueError('An enabled ROI requires rectangle bounds')
            return
        if not isinstance(self.bounds, (tuple, list)) or len(self.bounds) != 4:
            raise ValueError('ROI bounds must be x1,y1,x2,y2')
        bounds = tuple(_number(v, 'ROI coordinate') for v in self.bounds)
        x1, y1, x2, y2 = bounds
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('ROI must have positive area inside its reference frame')
        object.__setattr__(self, 'bounds', bounds)
        object.__setattr__(self, 'reference_size', (width, height))

    def validate_frame(self, frame_size):
        size = validate_frame_size(frame_size)
        if self.enabled and size != self.reference_size:
            raise ValueError('ROI reference size does not match the processing frame')

    def contains(self, box):
        if not self.enabled:
            return True
        x1, y1, x2, y2 = box
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2 if self.anchor == 'center' else y2
        left, top, right, bottom = self.bounds
        return left <= x <= right and top <= y <= bottom


@dataclass(frozen=True)
class DetectionConfig:
    confidence: float = 0.25
    inference_size: int = 640
    device: Optional[str] = None
    roi: RegionOfInterest = RegionOfInterest()

    def __post_init__(self):
        object.__setattr__(self, 'confidence', validate_probability(self.confidence))
        size = self.inference_size
        if isinstance(size, bool) or not isinstance(size, int) or size < 32 or size % 32:
            raise ValueError('Inference size must be a positive multiple of 32')
        if self.device is not None and (not isinstance(self.device, str) or
                (self.device not in ('cpu', 'mps') and not self.device.isdigit())):
            raise ValueError('Device must be cpu, mps, a GPU index such as 0, or None')
        if not isinstance(self.roi, RegionOfInterest):
            raise ValueError('ROI must be a RegionOfInterest configuration')

    def inference_arguments(self):
        arguments = {'conf': self.confidence, 'imgsz': self.inference_size, 'iou': NMS_IOU}
        if self.device is not None:
            arguments['device'] = self.device
        return arguments

    @classmethod
    def from_environment(cls, default=None, environ=None):
        """Read once before startup; never select sources through request input."""
        default = default or cls()
        env = os.environ if environ is None else environ
        changes = {}
        for variable, attribute, convert in (
                ('TRAFFIC_VEHICLE_CONFIDENCE', 'confidence', float),
                ('TRAFFIC_VEHICLE_IMGSZ', 'inference_size', int),
                ('TRAFFIC_VEHICLE_DEVICE', 'device', str)):
            if variable in env:
                try:
                    changes[attribute] = convert(env[variable])
                except (TypeError, ValueError) as error:
                    raise ValueError('Invalid ' + variable) from error
        roi_changes = {}
        if 'TRAFFIC_ROI_ENABLED' in env:
            value = env['TRAFFIC_ROI_ENABLED'].lower()
            if value not in ('true', 'false', '1', '0'):
                raise ValueError('TRAFFIC_ROI_ENABLED must be true, false, 1, or 0')
            roi_changes['enabled'] = value in ('true', '1')
        if 'TRAFFIC_ROI' in env:
            try:
                roi_changes['bounds'] = tuple(float(v) for v in env['TRAFFIC_ROI'].split(','))
            except (TypeError, ValueError) as error:
                raise ValueError('TRAFFIC_ROI must be x1,y1,x2,y2') from error
        if 'TRAFFIC_ROI_ANCHOR' in env:
            roi_changes['anchor'] = env['TRAFFIC_ROI_ANCHOR']
        if roi_changes:
            changes['roi'] = replace(default.roi, **roi_changes)
        return replace(default, **changes)


@dataclass
class DetectionCounts:
    returned_by_yolo: int = 0
    invalid: int = 0
    unsupported_class: int = 0
    below_confidence: int = 0
    outside_roi: int = 0
    accepted: int = 0


def prepare_detections(rows, frame_size, config, counts=None):
    """Validate each returned [xyxy, score, class] row; return float (N, 5).

    YOLO applies confidence before NMS. This boundary additionally rejects bad or
    below-threshold returned rows (e.g. alternate backends/test doubles). There
    is one application filter; there is no extra NMS. Exactly-at-threshold rows
    are accepted here; the installed YOLO's own strict NMS cutoff is preserved.
    Counts cannot observe candidates already discarded inside YOLO.
    """
    width, height = validate_frame_size(frame_size)
    config.roi.validate_frame((width, height))
    counts = counts if counts is not None else DetectionCounts()
    accepted = []
    for row in rows:
        counts.returned_by_yolo += 1
        try:
            if len(row) != 6:
                raise ValueError('Expected six values')
            x1, y1, x2, y2, score, class_id = (_number(v, 'Detection value') for v in row)
            validate_probability(score)
            if not class_id.is_integer() or x2 <= x1 or y2 <= y1:
                raise ValueError('Invalid class ID or box area')
            x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
            y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                raise ValueError('Box has no area after clipping')
        except (TypeError, ValueError, OverflowError):
            counts.invalid += 1
            continue
        if int(class_id) not in VEHICLE_CLASSES:
            counts.unsupported_class += 1
        elif score < config.confidence:
            counts.below_confidence += 1
        elif not config.roi.contains((x1, y1, x2, y2)):
            counts.outside_roi += 1
        else:
            accepted.append([x1, y1, x2, y2, score])
            counts.accepted += 1
    return np.asarray(accepted, dtype=float).reshape(-1, 5)


def detect_vehicles(model, frame, config, counts=None):
    result = model(frame, **config.inference_arguments())[0]
    # Ultralytics already maps xyxy back to the input image. Scaling again here
    # would corrupt the calibrated crossing geometry when imgsz changes.
    height, width = frame.shape[:2]
    return prepare_detections(result.boxes.data.tolist(), (width, height), config, counts)


def project_path(value, root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def resolve_video_source(value, root):
    """Local file or explicit webcam:N. Network sources are out of scope."""
    if isinstance(value, str) and value.startswith('webcam:'):
        index = value[len('webcam:'):]
        if not index.isascii() or not index.isdigit():
            raise ValueError('Webcam source must be webcam:N with a nonnegative integer index')
        return int(index)
    if not isinstance(value, (str, Path)) or not str(value).strip() or '://' in str(value):
        raise ValueError('Video source must be a local file or webcam:N')
    return project_path(value, root)
