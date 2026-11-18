"""Finite stop-line crossings in processing-image coordinates.

Candidates are observations, not legally confirmed violations. The manual signal
is a simulation; it is not synchronized with the original prerecorded scene.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
from uuid import uuid4


DIRECTIONS = {'down': (0, 1), 'up': (0, -1), 'right': (1, 0), 'left': (-1, 0)}


def _point(value):
    if len(value) != 2:
        raise ValueError('A point must have two coordinates')
    point = tuple(float(coordinate) for coordinate in value)
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError('Coordinates must be finite')
    return point


@dataclass(frozen=True)
class CrossingConfig:
    """Initial pixel/frame thresholds; calibrate these against labeled footage."""

    line_start: Tuple[float, float] = (160, 320)
    line_end: Tuple[float, float] = (550, 320)
    prohibited_direction: str = 'down'
    anchor_offset: float = 30
    hysteresis_pixels: float = 2
    endpoint_tolerance_pixels: float = 2
    minimum_movement_pixels: float = 2
    maximum_jump_pixels: float = 80
    maximum_frame_gap: int = 3
    history_expiration_frames: int = 90
    frame_size: Tuple[int, int] = (720, 480)

    def __post_init__(self):
        start, end = _point(self.line_start), _point(self.line_end)
        object.__setattr__(self, 'line_start', start)
        object.__setattr__(self, 'line_end', end)
        if start == end:
            raise ValueError('Stop-line endpoints must be different')
        if self.prohibited_direction not in DIRECTIONS:
            raise ValueError('Prohibited direction must be down, up, left, or right')
        width, height = self.frame_size
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
            raise ValueError('Frame size must contain positive integers')
        for point in (start, end):
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError('Stop line must lie inside the processing image')
        for name in ('anchor_offset', 'hysteresis_pixels', 'endpoint_tolerance_pixels',
                     'minimum_movement_pixels', 'maximum_jump_pixels'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if self.minimum_movement_pixels <= 0 or self.maximum_jump_pixels <= 0:
            raise ValueError('Movement and jump thresholds must be positive')
        if self.maximum_jump_pixels < self.minimum_movement_pixels:
            raise ValueError('Maximum jump must be at least the movement threshold')
        for name in ('maximum_frame_gap', 'history_expiration_frames'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if self.history_expiration_frames < self.maximum_frame_gap:
            raise ValueError('History expiration must not be shorter than the observation gap')
        dx, dy = end[0] - start[0], end[1] - start[1]
        direction = DIRECTIONS[self.prohibited_direction]
        if abs(-dy * direction[0] + dx * direction[1]) < 1e-9:
            raise ValueError('Prohibited direction cannot be parallel to the stop line')


def crossing_anchor(bbox, config):
    """Preserve integer bottom-center minus 30 px by default.

    The offset approximates a crossing point for the existing camera. It is not
    a calibrated tire/ground contact point; an offset of zero is bottom-center.
    All geometry uses the resized image, never the original video resolution.
    """
    if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
        raise ValueError('A bounding box must contain four finite coordinates')
    x1, y1, x2, y2 = map(int, bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Bounding box must have positive area')
    anchor = (int(x1 + (x2 - x1) / 2), int(y2 - config.anchor_offset))
    width, height = config.frame_size
    if not (0 <= anchor[0] < width and 0 <= anchor[1] < height):
        raise ValueError('Crossing anchor is outside the processing image')
    return anchor


def segment_intersection(previous, current, start, end, endpoint_tolerance=0):
    """Return (intersection, movement fraction), or None; supports sloped lines.

    Tolerance extends only the stop-line endpoints, not the observed movement.
    Parallel and collinear movement does not establish a directional crossing.
    """
    previous, current, start, end = map(_point, (previous, current, start, end))
    if not math.isfinite(endpoint_tolerance) or endpoint_tolerance < 0:
        raise ValueError('Endpoint tolerance must be finite and nonnegative')
    rx, ry = current[0] - previous[0], current[1] - previous[1]
    sx, sy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(sx, sy)
    if length == 0:
        raise ValueError('Stop line has zero length')
    denominator = rx * sy - ry * sx
    if abs(denominator) < 1e-9:
        return None
    ax, ay = start[0] - previous[0], start[1] - previous[1]
    fraction = (ax * sy - ay * sx) / denominator
    line_fraction = (ax * ry - ay * rx) / denominator
    tolerance = endpoint_tolerance / length
    if not (-1e-9 <= fraction <= 1 + 1e-9 and
            -tolerance - 1e-9 <= line_fraction <= 1 + tolerance + 1e-9):
        return None
    fraction = min(1.0, max(0.0, fraction))
    return ((previous[0] + fraction * rx, previous[1] + fraction * ry), fraction)


@dataclass
class TrackHistory:
    current_position: Tuple[float, float]
    current_frame_index: int
    last_seen_frame: int
    signal_state: str
    signal_version: int
    timestamp_ms: Optional[float]
    previous_position: Optional[Tuple[float, float]] = None
    previous_frame_index: Optional[int] = None
    approach_position: Optional[Tuple[float, float]] = None
    pending_crossing: Optional[dict] = None
    crossing_recorded: bool = False
    status: str = 'normal'
    reason: Optional[str] = None


class CrossingDetector:
    """One video session: short position history plus persistent crossing records.

    Transient histories expire. Crossing records/events persist until the session
    is replaced, so their memory use grows with unique crossings, not frames.
    Deduplication is by SORT ID and cannot resolve physical-vehicle ID switches.
    The Flask integration serializes access with its existing resource lock.
    """

    def __init__(self, config=None):
        self.config = config or CrossingConfig()
        self.history = {}
        self.recorded_crossings = {}
        self.events = {}
        self._last_frame_index = -1
        dx = self.config.line_end[0] - self.config.line_start[0]
        dy = self.config.line_end[1] - self.config.line_start[1]
        length = math.hypot(dx, dy)
        normal = (-dy / length, dx / length)
        self.direction = DIRECTIONS[self.config.prohibited_direction]
        if normal[0] * self.direction[0] + normal[1] * self.direction[1] < 0:
            normal = (-normal[0], -normal[1])
        self.normal = normal

    def _distance(self, point):
        return sum((point[i] - self.config.line_start[i]) * self.normal[i] for i in (0, 1))

    def _approaching(self, point):
        distance = self._distance(point)
        return distance < 0 and distance <= -self.config.hysteresis_pixels

    @staticmethod
    def _track_id(value):
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or number <= 0 or not number.is_integer():
            raise ValueError('Track ID must be a positive integer')
        return int(number)

    def expire(self, frame_index):
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < self._last_frame_index:
            raise ValueError('Frame indices must be nonnegative and monotonic')
        if frame_index < 0:
            raise ValueError('Frame indices must be nonnegative')
        if frame_index == self._last_frame_index:
            return
        self._last_frame_index = frame_index
        expired = [key for key, state in self.history.items()
                   if frame_index - state.last_seen_frame > self.config.history_expiration_frames]
        for key in expired:
            del self.history[key]

    def mark_uncertain(self, track_id, reason):
        try:
            track_id = self._track_id(track_id)
            self.history.pop(track_id, None)
        except (TypeError, ValueError, OverflowError):
            track_id = None
        return {'status': 'uncertain', 'reason': reason, 'track_id': track_id,
                'anchor': None, 'event': None}

    def observe_bbox(self, track_id, bbox, frame_index, signal_state,
                     signal_version=0, timestamp_ms=None):
        try:
            anchor = crossing_anchor(bbox, self.config)
        except (TypeError, ValueError, OverflowError) as error:
            return self.mark_uncertain(track_id, str(error))
        return self.update(track_id, anchor, frame_index, signal_state, signal_version, timestamp_ms)

    def update(self, track_id, position, frame_index, signal_state,
               signal_version=0, timestamp_ms=None):
        self.expire(frame_index)
        try:
            track_id = self._track_id(track_id)
            position = _point(position)
            width, height = self.config.frame_size
            if not (0 <= position[0] < width and 0 <= position[1] < height):
                raise ValueError('Position is outside the processing image')
        except (TypeError, ValueError, OverflowError) as error:
            return self.mark_uncertain(track_id, str(error))
        if timestamp_ms is not None:
            timestamp_ms = float(timestamp_ms)
            if not math.isfinite(timestamp_ms) or timestamp_ms < 0:
                timestamp_ms = None
        old = self.history.get(track_id)
        state = TrackHistory(position, frame_index, frame_index, signal_state,
                             signal_version, timestamp_ms)
        self.history[track_id] = state
        result = {'status': 'normal', 'reason': None, 'track_id': track_id,
                  'anchor': position, 'event': None}

        def finish(status, reason=None, event=None):
            state.status, state.reason = status, reason
            result.update(status=status, reason=reason, event=event)
            return result

        state.crossing_recorded = track_id in self.recorded_crossings
        if old is None:
            if self._approaching(position):
                state.approach_position = position
            if state.crossing_recorded:
                return finish(self.recorded_crossings[track_id]['status'])
            return finish('uncertain', 'first observation; movement is unknown')

        state.previous_position = old.current_position
        state.previous_frame_index = old.current_frame_index
        if state.crossing_recorded:
            return finish(self.recorded_crossings[track_id]['status'])
        gap = frame_index - old.current_frame_index
        movement = math.hypot(position[0] - old.current_position[0],
                              position[1] - old.current_position[1])
        if gap < 1 or gap > self.config.maximum_frame_gap or movement > self.config.maximum_jump_pixels:
            if self._approaching(position):
                state.approach_position = position
            return finish('uncertain', 'observation gap or position jump exceeds the configured limit')

        state.approach_position = old.approach_position
        state.pending_crossing = old.pending_crossing
        distance, previous_distance = self._distance(position), self._distance(old.current_position)
        if self._approaching(position):
            state.approach_position = position
        if distance < 0:
            state.pending_crossing = None

        intersection = segment_intersection(old.current_position, position,
                                            self.config.line_start, self.config.line_end,
                                            self.config.endpoint_tolerance_pixels)
        if previous_distance > 0 >= distance and intersection is not None:
            return finish('observed_crossing', 'opposite direction')
        if previous_distance < 0 <= distance and intersection is not None:
            segment_direction = sum((position[i] - old.current_position[i]) * self.direction[i]
                                    for i in (0, 1))
            if segment_direction <= 0:
                state.pending_crossing = None
                return finish('observed_crossing', 'movement is not in the prohibited direction')
            if state.approach_position is None:
                return finish('uncertain', 'no observation clearly before the stop line')
            point, fraction = intersection
            timestamp = None
            if old.timestamp_ms is not None and timestamp_ms is not None and timestamp_ms > old.timestamp_ms:
                timestamp = old.timestamp_ms + fraction * (timestamp_ms - old.timestamp_ms)
            consistent_signal = (old.signal_state == signal_state and old.signal_version == signal_version)
            state.pending_crossing = {
                'track_id': track_id, 'frame_index': frame_index,
                'timestamp_ms': timestamp, 'crossing_point': list(point),
                'signal_state': signal_state, 'signal_consistent': consistent_signal,
            }

        pending = state.pending_crossing
        if pending is None:
            return finish('normal')
        approach = state.approach_position
        directed_movement = sum((position[i] - approach[i]) * self.direction[i] for i in (0, 1))
        if distance < self.config.hysteresis_pixels or directed_movement < self.config.minimum_movement_pixels:
            return finish('uncertain', 'crossing awaits movement beyond the hysteresis band')

        # Consume the crossing even if green/unknown: a later light change must
        # never retroactively turn this same movement into a red-light event.
        state.crossing_recorded = True
        state.pending_crossing = None
        record = dict(pending)
        if not pending['signal_consistent'] or pending['signal_state'] not in ('red', 'green'):
            record['status'] = 'uncertain'
            self.recorded_crossings[track_id] = record
            return finish('uncertain', 'unknown signal or signal transition during the crossing interval')
        record['status'] = 'observed_crossing'
        self.recorded_crossings[track_id] = record
        if pending['signal_state'] != 'red':
            return finish('observed_crossing', 'crossed during green')

        event = {key: pending[key] for key in ('track_id', 'frame_index', 'timestamp_ms',
                                              'crossing_point', 'signal_state')}
        event.update(event_id=uuid4().hex, status='candidate', license_plate=None,
                     license_plate_confidence=None)
        self.events[track_id] = event
        record['status'] = 'candidate'
        return finish('candidate', event=event)

    def enrich_event(self, track_id, license_plate, confidence):
        """Attach valid OCR once/better; never promote a candidate to confirmed."""
        event = self.events.get(track_id)
        if event is None or not isinstance(license_plate, str) or not license_plate.strip():
            return False
        try:
            confidence = float(confidence)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return False
        previous_confidence = event['license_plate_confidence']
        if previous_confidence is not None and confidence <= previous_confidence:
            return False
        event['license_plate'] = license_plate
        event['license_plate_confidence'] = confidence
        return True
