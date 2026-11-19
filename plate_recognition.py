"""Conservative, model-independent plate recognition and bounded event evidence.

Syntax acceptance is not registration verification. Scores are engine scores,
not calibrated probabilities. No partial strings or different IDs are joined.
"""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
import math
from itertools import islice
from numbers import Real
import os
import re
from typing import Optional, Tuple

import cv2
import numpy as np


STATES = frozenset('AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY RJ SK TG TN TR TS UK UP UT WB'.split())
TO_DIGIT = dict(zip('OIZSBG', '012586'))
TO_LETTER = {value: key for key, value in TO_DIGIT.items()}
METHODS = ('color', 'gray', 'fixed', 'otsu', 'adaptive', 'contrast')


def probability(value):
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, Real)
            and math.isfinite(value) and 0 <= value <= 1)


@dataclass(frozen=True)
class OCRConfig:
    plate_confidence: float = .25
    ocr_confidence: float = .65
    single_read_confidence: float = .90
    min_observations: int = 2
    history_size: int = 12
    history_frames: int = 90
    interval_frames: int = 1
    max_backoff: int = 8
    max_attempts: int = 20
    state_ttl: int = 150
    max_states: int = 128
    min_width: int = 16
    min_height: int = 8
    max_corrections: int = 2
    max_candidates: int = 8
    preprocessing: str = 'fixed'
    upscale: float = 1.0
    allow_legacy: bool = True
    extra_patterns: Tuple[Tuple[str, str], ...] = ()
    diagnostic: bool = False

    def __post_init__(self):
        for key in ('plate_confidence', 'ocr_confidence', 'single_read_confidence'):
            if not probability(getattr(self, key)):
                raise ValueError(key + ' must be finite in [0,1]')
        if self.single_read_confidence < self.ocr_confidence:
            raise ValueError('single_read_confidence must be >= ocr_confidence')
        for key in ('min_observations', 'history_size', 'history_frames', 'interval_frames',
                    'max_backoff', 'max_attempts', 'state_ttl', 'max_states', 'min_width',
                    'min_height', 'max_candidates'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(key + ' must be a positive integer')
        if self.min_observations > self.history_size:
            raise ValueError('min_observations exceeds history_size')
        if isinstance(self.max_corrections, bool) or not isinstance(self.max_corrections, int) or not 0 <= self.max_corrections <= 3:
            raise ValueError('max_corrections must be an integer in [0,3]')
        if self.preprocessing not in METHODS:
            raise ValueError('Unsupported preprocessing')
        if isinstance(self.upscale, bool) or not isinstance(self.upscale, Real) or not math.isfinite(self.upscale) or not 1 <= self.upscale <= 3:
            raise ValueError('upscale must be finite in [1,3]')
        if not isinstance(self.allow_legacy, bool) or not isinstance(self.diagnostic, bool):
            raise ValueError('Boolean flags require bool values')
        for name, pattern in self.extra_patterns:
            if not isinstance(name, str) or not name or not isinstance(pattern, str):
                raise ValueError('Extra patterns require (name, regex) strings')
            re.compile(pattern, re.ASCII)

    @classmethod
    def from_environment(cls, default=None, environ=None):
        env = os.environ if environ is None else environ
        values = {}
        integer = ('min_observations', 'history_size', 'history_frames', 'interval_frames',
                   'max_backoff', 'max_attempts', 'state_ttl', 'max_states', 'min_width',
                   'min_height', 'max_corrections', 'max_candidates')
        for name in integer + ('plate_confidence', 'ocr_confidence', 'single_read_confidence', 'upscale', 'preprocessing', 'allow_legacy', 'diagnostic'):
            key = ('TRAFFIC_PLATE_CONFIDENCE' if name == 'plate_confidence' else
                   'TRAFFIC_OCR_CONFIDENCE' if name == 'ocr_confidence' else
                   'TRAFFIC_OCR_' + name.upper())
            if key not in env:
                continue
            try:
                if name in ('allow_legacy', 'diagnostic'):
                    if env[key].lower() not in ('true', 'false'):
                        raise ValueError('Expected true or false')
                    value = env[key].lower() == 'true'
                else:
                    value = (int if name in integer else str if name == 'preprocessing' else float)(env[key])
                values[name] = value
            except (ValueError, TypeError) as error:
                raise ValueError('Invalid ' + key) from error
        return replace(default or cls(), **values)


def normalize_text(text):
    """Remove only whitespace and common separators; retain unsupported symbols.

    Reject non-ASCII text rather than letting uppercase expand e.g. sharp-s.
    Punctuation is retained so validation cannot silently turn noise into a plate.
    """
    if not isinstance(text, str) or not text.isascii():
        return ''
    return re.sub(r'[\s\-]', '', text.upper()) if len(text) <= 64 else ''


@dataclass(frozen=True)
class Validation:
    normalized_text: str
    format_type: Optional[str]
    is_valid: bool


def validate_plate(text, config=None):
    config = config or OCRConfig()
    value = normalize_text(text)
    kind = None
    if re.fullmatch(r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}', value) and value[:2] in STATES:
        kind = 'standard'
    elif re.fullmatch(r'[0-9]{2}BH[0-9]{4}[A-Z]{2}', value):
        kind = 'bharat'
    elif config.allow_legacy and re.fullmatch(r'[A-Z]{2}[0-9]{2}[A-Z]{3}', value):
        kind = 'legacy'
    else:
        for name, pattern in config.extra_patterns:
            if re.fullmatch(pattern, value, re.ASCII):
                kind = name
                break
    return Validation(value, kind, kind is not None)


@dataclass(frozen=True)
class Candidate:
    original_text: str
    normalized_text: str
    format_type: str
    confidence: float
    corrections: int = 0


def correction_candidates(text, confidence, config=None):
    config = config or OCRConfig()
    if not probability(confidence):
        return ()
    result = validate_plate(text, config)
    if result.is_valid:
        return (Candidate(text, result.normalized_text, result.format_type, float(confidence)),)
    value = result.normalized_text
    if not value or not re.fullmatch('[A-Z0-9]+', value):
        return ()
    masks = [('standard', 'LL' + 'D' * district + 'L' * series + 'DDDD')
             for district in (1, 2) for series in (1, 2, 3)]
    masks += [('bharat', 'DD' + 'BH' + 'DDDDLL')]  # literal B/H positions are never guessed
    if config.allow_legacy:
        masks.append(('legacy', 'LLDDLLL'))
    candidates = {}
    for kind, mask in masks:
        if len(mask) != len(value):
            continue
        output, edits = [], 0
        for char, expected in zip(value, mask):
            converted = (TO_LETTER.get(char, char) if expected == 'L' else
                         TO_DIGIT.get(char, char) if expected == 'D' else char)
            if expected not in ('L', 'D') and char != expected:
                break
            output.append(converted)
            edits += converted != char
        if len(output) != len(value) or not 0 < edits <= config.max_corrections:
            continue
        checked = validate_plate(''.join(output), config)
        if checked.is_valid and checked.format_type == kind:
            candidates[checked.normalized_text] = Candidate(text, checked.normalized_text,
                                                          checked.format_type, float(confidence), edits)
    # Exceeding the cap is a rejection, never permission to hide alternatives.
    if len(candidates) > config.max_candidates:
        return ()
    return tuple(sorted(candidates.values(), key=lambda c: (c.corrections, c.normalized_text)))


@dataclass(frozen=True)
class Reading:
    status: str
    candidates: tuple = ()
    counts: dict = field(default_factory=dict)

    @property
    def best(self):
        return self.candidates[0] if self.candidates and self.status == 'pending' else None

    def __iter__(self):
        # Existing callers may still unpack (text, score).
        yield self.best.normalized_text if self.best else None
        yield self.best.confidence if self.best else None


def rank_ocr(detections, config=None):
    config = config or OCRConfig()
    counts = Counter()
    candidates = {}
    rows = [] if detections is None else list(islice(detections,64))
    if not rows:
        return Reading('unreadable', counts={'empty_results': 1})
    for row in rows:
        counts['ocr_results'] += 1
        if not isinstance(row, (list, tuple)) or len(row) != 3 or not probability(row[2]):
            counts['invalid_confidence_results'] += 1
            continue
        original, score = row[1], row[2]
        choices = correction_candidates(original, score, config)
        if not choices:
            counts['format_rejected_results'] += 1
            continue
        if score < config.ocr_confidence:
            counts['low_confidence_results'] += 1
            continue
        # Ambiguous format positions from ONE text are not independent evidence.
        if len(choices) > 1:
            counts['ambiguous_corrections'] += 1
            continue
        candidate = choices[0]
        old = candidates.get(candidate.normalized_text)
        if old is None or (candidate.corrections, -candidate.confidence) < (old.corrections, -old.confidence):
            candidates[candidate.normalized_text] = candidate
    ordered = tuple(sorted(candidates.values(), key=lambda c: (c.corrections, -c.confidence, c.normalized_text))[:config.max_candidates])
    if ordered:
        ambiguous = (len(ordered) > 1 and ordered[0].corrections == ordered[1].corrections
                     and ordered[0].confidence - ordered[1].confidence < .05)
        return Reading('conflicting' if ambiguous else 'pending', ordered, dict(counts))
    status = ('low_confidence' if counts['low_confidence_results'] else
              'conflicting' if counts['ambiguous_corrections'] else 'format_unrecognized')
    return Reading(status, counts=dict(counts))


def coerce_reading(value, config):
    if isinstance(value, Reading):
        return value
    # Compatibility for callers providing the established (text, confidence) API.
    text, confidence = value
    return rank_ocr([] if text is None else [(None, text, confidence)], config)


def valid_box(box):
    try:
        if len(box) != 4 or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in box):
            return None
        x1, y1, x2, y2 = map(float, box)
        if not all(math.isfinite(x) for x in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2
    except (TypeError, ValueError, OverflowError):
        return None


def safe_crop(image, box, config=None):
    config = config or OCRConfig()
    bounds = valid_box(box)
    if bounds is None or not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or not image.size:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bounds
    x1, y1 = max(0, min(width, math.floor(x1))), max(0, min(height, math.floor(y1)))
    x2, y2 = max(0, min(width, math.ceil(x2))), max(0, min(height, math.ceil(y2)))
    if x2 - x1 < config.min_width or y2 - y1 < config.min_height:
        return None
    return image[y1:y2, x1:x2].copy()


def associate_plate(plate, vehicles, minimum_containment=.7, ambiguity_margin=.12):
    """Center + >=70% contained area, modest plate/vehicle area, score margin.

    Score uses containment and relative plate area only; near ties are unassigned.
    Duplicate IDs are rejected rather than picking a row. No nearest/OCR fallback.
    """
    try:
        p = valid_box(plate[:4])
        rows = [row for row in vehicles if isinstance(row, (tuple,list,np.ndarray)) and len(row) == 5]
    except (TypeError, ValueError):
        return (-1,) * 5
    if p is None:
        return (-1,) * 5
    x1, y1, x2, y2 = p
    area = (x2-x1)*(y2-y1)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    ids = Counter(row[4] for row in rows if len(row) == 5 and probability_id(row[4]))
    ranked = []
    for row in rows:
        if len(row) != 5 or not probability_id(row[4]) or ids[row[4]] != 1:
            continue
        v = valid_box(row[:4])
        if v is None:
            continue
        a,b,c,d = v
        if not a <= cx <= c or not b <= cy <= d:
            continue
        fraction = max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))/area
        relative = area/((c-a)*(d-b))
        if fraction < minimum_containment or relative > .4:
            continue
        score = .8*fraction + .2*min(relative/.1, 1)
        ranked.append((score, int(row[4]), tuple(row)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    if not ranked or (len(ranked)>1 and ranked[0][0]-ranked[1][0] < ambiguity_margin):
        return (-1,) * 5
    return ranked[0][2]


def probability_id(value):
    return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(value) and 0 < value <= 2**53-1 and float(value).is_integer())


def preprocess(crop, method='fixed', upscale=1.0):
    if method not in METHODS or isinstance(upscale, bool) or not isinstance(upscale, Real) or not math.isfinite(upscale) or not 1 <= upscale <= 3:
        raise ValueError('Invalid preprocessing configuration')
    if not isinstance(crop, np.ndarray) or crop.dtype != np.uint8 or not crop.size or crop.ndim not in (2,3) or (crop.ndim == 3 and crop.shape[2] != 3):
        raise ValueError('Expected nonempty uint8 gray or BGR crop')
    image = crop.copy()
    if upscale != 1:
        image = cv2.resize(image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    if method == 'color':
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if method == 'gray':
        return gray
    if method == 'fixed':
        return cv2.threshold(gray,64,255,cv2.THRESH_BINARY_INV)[1]
    if method == 'otsu':
        return cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    if method == 'adaptive':
        return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)


@dataclass
class EventEvidence:
    history: deque
    attempts: int = 0
    last_attempt: int = -1000000
    last_seen: int = -1
    best_quality: float = 0


class OCRManager:
    """Bounded synchronous state keyed by event UUID, never shared across IDs."""
    def __init__(self, config=None):
        self.config = config or OCRConfig()
        self.states = {}
        self.counts = Counter()
        self.ocr_seconds = 0.0
        self.preprocessing_seconds = 0.0

    def metadata(self, event):
        return event.setdefault('recognition', {'status':'not_attempted', 'attempts':0,
                                               'observations':0, 'reason':None})

    def expire(self, frame, events):
        by_id = {event['event_id']:event for event in events.values()}
        for key, state in list(self.states.items()):
            self._trim(state, frame)
            if frame-state.last_seen > self.config.state_ttl or key not in by_id:
                self.states.pop(key)
                if key in by_id:
                    self.metadata(by_id[key]).update(status='unreadable', reason='observation_expired', closed=True)

    def invalidate(self, event, reason='identity_uncertain'):
        self.states.pop(event['event_id'], None)
        meta = self.metadata(event)
        meta.update(status='accepted' if event.get('license_plate') else 'conflicting', reason=reason, closed=True)

    def due(self, event, frame, quality=0, scan=False):
        meta = self.metadata(event)
        if meta.get('closed') or meta['status'] == 'accepted' or event.get('license_plate'):
            return False
        state = self.states.get(event['event_id'])
        if state is None:
            if len(self.states) >= self.config.max_states:
                meta.update(status='pending', reason='capacity_limit')
                return False
            state = EventEvidence(deque(maxlen=self.config.history_size), attempts=meta['attempts'])
            self.states[event['event_id']] = state
        state.last_seen = frame
        if state.attempts >= self.config.max_attempts:
            meta.update(reason='attempt_limit', closed=True)
            self.states.pop(event['event_id'], None)
            return False
        delay = self.config.interval_frames * min(max(state.attempts,1), self.config.max_backoff)
        better = quality > 0 and quality >= state.best_quality * 1.5
        return frame-state.last_attempt >= (self.config.interval_frames if better or scan else delay)

    def _trim(self, state, frame):
        while state.history and frame-state.history[0]['frame_index'] > self.config.history_frames:
            state.history.popleft()

    def begin(self, event, frame, quality):
        state = self.states[event['event_id']]
        state.attempts += 1
        state.last_attempt = frame
        state.best_quality = max(state.best_quality, quality)
        self.metadata(event).update(status='pending', attempts=state.attempts, reason=None)
        self.counts['ocr_attempts'] += 1

    def record(self, event, frame, reading, plate_score):
        reading = coerce_reading(reading, self.config)
        self.counts.update(reading.counts)
        meta = self.metadata(event)
        if meta.get('closed') or meta['status'] == 'accepted':
            return None
        state = self.states[event['event_id']]
        if any(item['frame_index'] == frame for item in state.history):
            self.counts['duplicate_observations'] += 1
            return None
        self._trim(state, frame)
        candidate = reading.best
        state.history.append({'frame_index':frame, 'original_text':candidate.original_text if candidate else None,
                              'normalized_text':candidate.normalized_text if candidate else None,
                              'ocr_confidence':candidate.confidence if candidate else None,
                              'plate_confidence':plate_score, 'status':reading.status,
                              'corrections':candidate.corrections if candidate else None})
        if not candidate:
            meta.update(status=reading.status, reason=reading.status)
            return None
        grouped = defaultdict(list)
        for observation in state.history:
            if observation['normalized_text']:
                grouped[observation['normalized_text']].append(observation)
        votes = sorted(grouped.items(), key=lambda pair: (
            -sum(o['ocr_confidence'] * o['plate_confidence'] / (1+.2*o['corrections']) for o in pair[1]), pair[0]))
        text, observations = votes[0]
        weight = lambda group: sum(o['ocr_confidence']*o['plate_confidence']/(1+.2*o['corrections']) for o in group)
        conflict = len(votes)>1 and weight(observations) <= 1.5*weight(votes[1][1])
        confidence = max(o['ocr_confidence'] for o in observations)
        single_strong = (len(votes)==1 and any(o['corrections']==0 and o['ocr_confidence'] >= self.config.single_read_confidence for o in observations))
        accepted = not conflict and (single_strong or len(observations) >= self.config.min_observations)
        meta.update(status='accepted' if accepted else 'conflicting' if conflict else 'pending',
                    provisional_text=text, confidence=confidence, observations=len(observations),
                    corrected=any(o['corrections'] for o in observations))
        if accepted:
            self.counts['accepted_results'] += 1
            self.states.pop(event['event_id'], None)
            return text, confidence
        return None

    def clear(self):
        self.states.clear()
