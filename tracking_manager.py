"""Validated SORT configuration, bounded lifecycle state and observed outputs.

SORT owns motion prediction and association. This adapter never guesses that
two IDs are one vehicle, changes Kalman equations, or owns violation events.
"""

from dataclasses import dataclass, replace
from collections import Counter
import math
from numbers import Real
import os
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TrackingConfig:
    max_age: int = 1
    min_hits: int = 3
    iou_threshold: float = 0.3

    def __post_init__(self):
        for name, minimum in (('max_age', 0), ('min_hits', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(name + ' must be an integer >= ' + str(minimum))
        value = self.iou_threshold
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('iou_threshold must be finite and between 0 and 1')

    @classmethod
    def from_environment(cls, default=None, environ=None):
        env = os.environ if environ is None else environ
        values = {}
        for key, name, convert in (('TRAFFIC_SORT_MAX_AGE', 'max_age', int),
                                   ('TRAFFIC_SORT_MIN_HITS', 'min_hits', int),
                                   ('TRAFFIC_SORT_IOU_THRESHOLD', 'iou_threshold', float)):
            if key in env:
                try:
                    values[name] = convert(env[key])
                except (TypeError, ValueError) as error:
                    raise ValueError('Invalid ' + key) from error
        return replace(default or cls(), **values)


def _array(rows, columns):
    # Reject nonnumeric values rather than coercing strings/bools into geometry.
    if not isinstance(rows, np.ndarray):
        objects = np.asarray(rows, dtype=object)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in objects.flat):
            raise ValueError('Tracking arrays must contain real numbers, not strings or booleans')
    raw = np.asarray(rows)
    if not raw.size:
        if raw.shape not in ((0,), (0, columns)):
            raise ValueError('Expected an (N, {}) array'.format(columns))
        return np.empty((0, columns), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != columns or raw.dtype.kind not in 'iuf':
        raise ValueError('Expected a numeric (N, {}) array'.format(columns))
    return np.asarray(raw, dtype=float)


def _box(values, frame_size):
    box = np.asarray(values, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return None
    width, height = frame_size
    box = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return tuple(float(value) for value in box)


def validate_detections(rows, frame_size=(720, 480)):
    """A programming error at this boundary fails before SORT state advances."""
    array = _array(rows, 5)
    valid = []
    for row in array:
        box = _box(row[:4], frame_size)
        if box is None or not math.isfinite(row[4]) or not 0 <= row[4] <= 1:
            raise ValueError('SORT input requires finite positive-area boxes and confidence in [0,1]')
        valid.append([*box, row[4]])
    return np.asarray(valid, dtype=float).reshape(-1, 5)


def sanitize_tracks(rows, frame_size=(720, 480)):
    """Return (safe Nx5 tracks, rejected known IDs), excluding ALL duplicate IDs.

    Rejecting every ambiguous row avoids selecting one vehicle arbitrarily.
    Known rejected IDs can invalidate transient crossing history, not events.
    """
    array = _array(rows, 5)
    identities = [int(row[4]) if math.isfinite(row[4]) and 0 < row[4] <= 2**53 - 1
                  and row[4].is_integer() else None for row in array]
    frequencies = Counter(identity for identity in identities if identity is not None)
    valid, rejected = [], set()
    for row, identity in zip(array, identities):
        box = _box(row[:4], frame_size)
        if identity is None or frequencies[identity] > 1 or box is None:
            if identity is not None:
                rejected.add(identity)
            continue
        valid.append([*box, identity])
    return np.asarray(valid, dtype=float).reshape(-1, 5), rejected


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: int
    first_seen_frame: int
    last_observed_frame: int
    missed_frames: int
    confirmed: bool
    bbox: Optional[Tuple[float, float, float, float]]
    observed: bool
    state: str


class TrackingManager:
    """Confirm after min_hits consecutive associations (creation is not a hit).

    Suppress SORT's startup bypass. Once confirmed, retain confirmation during
    permitted occlusions and return corrected states immediately on observation
    recovery. Unobserved predictions are metadata only, never crossing input.
    Live metadata is bounded by SORT's active tracks; expired/recovered IDs are
    one-update notifications. IDs are never reset or reassigned by this class.
    """

    def __init__(self, config=None, frame_size=(720, 480)):
        from vehicle_detector import validate_frame_size
        from sort.sort import Sort
        if config is not None and not isinstance(config, TrackingConfig):
            raise ValueError('config must be a TrackingConfig')
        self.config = config if config is not None else TrackingConfig()
        self.frame_size = validate_frame_size(frame_size)
        self.sort = Sort(max_age=self.config.max_age, min_hits=self.config.min_hits,
                         iou_threshold=self.config.iou_threshold)
        self.tracks = {}
        self._confirmed = set()
        self.expired_ids = ()
        self.recovered_ids = ()
        self.invalid_ids = set()
        self.created_count = 0
        self.expired_count = 0
        self.recovered_count = 0

    def update(self, detections):
        detections = validate_detections(detections, self.frame_size)
        self.sort.update(detections)
        frame = self.sort.frame_count - 1
        previous = self.tracks
        current, candidates, recovered = {}, [], []
        identities = Counter(track.id + 1 for track in self.sort.trackers)
        ambiguous = {identity for identity, count in identities.items() if count > 1}
        for track in self.sort.trackers:
            identity = track.id + 1
            if identity in ambiguous:
                continue
            observed = track.time_since_update == 0
            if observed and track.hit_streak >= self.config.min_hits:
                self._confirmed.add(identity)
            confirmed = identity in self._confirmed
            raw_box = track.get_state()[0]
            box = _box(raw_box, self.frame_size)
            current[identity] = TrackSnapshot(identity, frame - track.age,
                frame - track.time_since_update, track.time_since_update, confirmed, box,
                observed, 'lost' if not observed else ('confirmed' if confirmed else 'tentative'))
            if observed and identity in previous and not previous[identity].observed:
                recovered.append(identity)
            if confirmed and observed:
                candidates.append([*raw_box, identity])
        # Do not retain historical snapshots or expired confirmation flags.
        expired = previous.keys() - current.keys()
        self.created_count += len(current.keys() - previous.keys())
        self.expired_count += len(expired)
        self.recovered_count += len(recovered)
        self.expired_ids = tuple(sorted(expired))
        self.recovered_ids = tuple(sorted(recovered))
        self.tracks = current
        self._confirmed.intersection_update(current)
        rows, rejected = sanitize_tracks(candidates, self.frame_size)
        rejected.update(ambiguous)
        # Invalid tentative/lost states must also break old crossing observations.
        rejected.update(identity for identity, state in current.items() if state.bbox is None)
        self.invalid_ids = rejected
        return rows

    def clear(self):
        """Release live tracking state; violation history belongs to its caller."""
        self.expired_ids = tuple(sorted(self.tracks))
        self.expired_count += len(self.tracks)
        self.sort.trackers.clear()
        self.tracks.clear()
        self._confirmed.clear()
        self.recovered_ids = ()
        self.invalid_ids.clear()
