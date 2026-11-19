"""Bounded, redacted monitoring snapshots; never reads the database or images."""
from copy import deepcopy
from itertools import islice
import math
from threading import Lock
from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

RECOGNITION_STATUSES = ('not_attempted', 'pending', 'unreadable', 'accepted',
                        'conflicting', 'low_confidence', 'format_unrecognized')


class SignalStatus(BaseModel):
    color: Literal['red', 'green']
    version: int


class SourceStatus(BaseModel):
    kind: Literal['unavailable', 'local_video', 'camera'] = 'unavailable'
    width: Optional[int] = None
    height: Optional[int] = None
    total_frames: Optional[int] = None


class Metrics(BaseModel):
    processed_frames: Optional[int] = None
    frame_index: Optional[int] = None
    processing_fps: Optional[float] = None
    source_fps: Optional[float] = None
    vehicle_detections: Optional[int] = None
    observed_tracks: Optional[int] = None
    candidates: Optional[int] = None
    accepted_plates: Optional[int] = None
    active_viewers: int = 0


class CandidateSummary(BaseModel):
    track_id: int
    frame_index: int
    signal: Literal['red', 'green', 'unknown']
    status: Literal['candidate', 'uncertain']
    recognition: Literal['not_attempted', 'pending', 'unreadable', 'accepted',
                         'conflicting', 'low_confidence', 'format_unrecognized']


class StorageStatus(BaseModel):
    state: Literal['unavailable', 'disabled', 'ready', 'degraded'] = 'unavailable'
    evidence_capture: bool = False


class AccessStatus(BaseModel):
    history: bool = False
    evidence: bool = False


class DashboardStatus(BaseModel):
    schema_version: Literal[1] = 1
    state: Literal['not_started', 'initializing', 'ready', 'processing',
                   'completed', 'stopping', 'stopped', 'error']
    session: Optional[str] = None
    signal: SignalStatus
    source: SourceStatus
    metrics: Metrics
    recent_candidates: List[CandidateSummary] = Field(default_factory=list)
    storage: StorageStatus
    access: AccessStatus = Field(default_factory=AccessStatus)
    error: Optional[str] = None


class DashboardState:
    """One owner writes; requests copy at most 20 small current-session rows."""
    def __init__(self):
        self._lock = Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._data = dict(state='not_started', session=None, source={},
                              metrics={}, recent_candidates=[], storage={})
            self._seconds = 0.0
            self._detections = 0
            self._frames = 0

    def ready(self, source, frame_size, frame_count, storage):
        total = (int(frame_count) if isinstance(frame_count, (int, float))
                 and math.isfinite(frame_count) and frame_count > 0 else None)
        with self._lock:
            self._data.update(state='ready', session=uuid4().hex,
                source=dict(kind='camera' if isinstance(source, int) else 'local_video',
                            width=frame_size[0], height=frame_size[1], total_frames=total),
                storage=self.storage_status(storage))
            self._data['metrics'] = dict(processed_frames=0, frame_index=None,
                processing_fps=None, vehicle_detections=0, observed_tracks=0,
                candidates=0, accepted_plates=0, source_fps=None)

    @staticmethod
    def storage_status(storage):
        if storage is None:
            return dict(state='unavailable', evidence_capture=False)
        state = ('disabled' if not storage.config.enabled else
                 'degraded' if storage.counts.get('failures', 0) or not storage.session_ready else 'ready')
        return dict(state=state, evidence_capture=bool(storage.config.evidence_enabled))

    def frame(self, index, seconds, detections, tracks, events, accepted, fps, storage):
        # Most recent *created* candidates, not a complete history. No plate,
        # path, exception, image, persistence record or review note is copied.
        rows = []
        for event in islice(reversed(events.values()), 20):
            recognition = event.get('recognition', {}).get('status', 'not_attempted')
            rows.append(dict(track_id=event['track_id'], frame_index=event['frame_index'],
                signal=event['signal_state'] if event['signal_state'] in ('red', 'green') else 'unknown',
                status='uncertain' if event.get('status') == 'uncertain' else 'candidate',
                recognition=recognition if recognition in RECOGNITION_STATUSES else 'unreadable'))
        with self._lock:
            self._frames += 1
            self._seconds += max(0, seconds)
            self._detections += detections
            self._data.update(state='processing', recent_candidates=rows,
                              storage=self.storage_status(storage))
            self._data['metrics'] = dict(processed_frames=self._frames, frame_index=index,
                processing_fps=self._frames / self._seconds if self._seconds > 0 else None,
                vehicle_detections=self._detections, observed_tracks=tracks,
                candidates=len(events), accepted_plates=accepted, source_fps=fps)

    def finish(self, status, storage=None):
        with self._lock:
            if storage is not None:
                self._data['storage'] = self.storage_status(storage)
            # Cleanup after EOF must not erase the terminal outcome.
            if self._data['state'] not in ('completed', 'error'):
                self._data['state'] = {'failed':'error', 'completed':'completed'}.get(status, 'stopped')

    def snapshot(self):
        with self._lock:
            return deepcopy(self._data)
