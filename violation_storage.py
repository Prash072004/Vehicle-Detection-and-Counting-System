"""Storage configuration and bounded synchronous persistence integration.

Database calls are short and never wrap inference/OCR. Failed candidates remain
in the existing detector's memory with explicit persistence status and bounded
retry attempts. Evidence is an independent opt-in, never enabled by import.
"""
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from uuid import uuid4

from database import Database, StorageError, integer, utc_time
from evidence_manager import EvidenceManager
from plate_recognition import OCRConfig, validate_plate


ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StorageConfig:
    enabled: bool = True
    database_path: Path = ROOT/'data'/'violations.db'
    evidence_enabled: bool = False
    evidence_directory: Path = ROOT/'evidence'
    annotated_evidence: bool = False
    plate_evidence: bool = False
    buffer_frames: int = 5
    busy_timeout_ms: int = 200
    max_attempts: int = 3
    retry_interval_frames: int = 30
    max_pending_events: int = 128
    operations_per_frame: int = 2
    retention_days: int = 0

    def __post_init__(self):
        for key in ('enabled','evidence_enabled','annotated_evidence','plate_evidence'):
            if not isinstance(getattr(self,key),bool):
                raise ValueError(key+' must be boolean')
        if self.evidence_enabled and not self.enabled:
            raise ValueError('Evidence requires persistent storage')
        integer(self.buffer_frames,'buffer_frames',2,30)
        integer(self.busy_timeout_ms,'busy_timeout_ms',0,5000)
        integer(self.max_attempts,'max_attempts',1,10)
        integer(self.retry_interval_frames,'retry_interval_frames',1,10000)
        integer(self.max_pending_events,'max_pending_events',1,1000)
        integer(self.operations_per_frame,'operations_per_frame',1,10)
        integer(self.retention_days,'retention_days',0,36500)
        if Path(self.database_path).suffix.lower() not in ('.db','.sqlite','.sqlite3'):
            raise ValueError('Invalid database extension')

    @classmethod
    def from_environment(cls,default=None,environ=None):
        env = os.environ if environ is None else environ
        values = {}
        names = {'enabled':'TRAFFIC_STORAGE_ENABLED','database_path':'TRAFFIC_DATABASE_PATH',
                 'evidence_enabled':'TRAFFIC_EVIDENCE_ENABLED','evidence_directory':'TRAFFIC_EVIDENCE_DIR',
                 'annotated_evidence':'TRAFFIC_EVIDENCE_ANNOTATED','plate_evidence':'TRAFFIC_EVIDENCE_PLATES',
                 'buffer_frames':'TRAFFIC_EVIDENCE_BUFFER_FRAMES','busy_timeout_ms':'TRAFFIC_STORAGE_BUSY_TIMEOUT_MS',
                 'max_attempts':'TRAFFIC_STORAGE_MAX_ATTEMPTS','retry_interval_frames':'TRAFFIC_STORAGE_RETRY_FRAMES',
                 'max_pending_events':'TRAFFIC_STORAGE_MAX_PENDING','operations_per_frame':'TRAFFIC_STORAGE_OPERATIONS_PER_FRAME',
                 'retention_days':'TRAFFIC_RETENTION_DAYS'}
        for name,key in names.items():
            if key not in env:
                continue
            try:
                if name in ('enabled','evidence_enabled','annotated_evidence','plate_evidence'):
                    if env[key].lower() not in ('true','false'):
                        raise ValueError('Expected true/false')
                    value = env[key].lower()=='true'
                elif name in ('database_path','evidence_directory'):
                    value = Path(env[key])
                    value = (value if value.is_absolute() else ROOT/value).resolve()
                    allowed = (ROOT/'outputs',ROOT/('data' if name=='database_path' else 'evidence'))
                    if not any(value==base or base in value.parents for base in allowed):
                        raise ValueError('Storage paths must stay in data/evidence or outputs')
                else:
                    value = int(env[key])
                values[name] = value
            except (ValueError,TypeError,AttributeError) as error:
                raise ValueError('Invalid '+key) from error
        config = replace(default or cls(),**values)
        # Validate defaults too: an existing symlink/junction must not redirect
        # application storage into source, protected assets or outside the repo.
        paths = {}
        for name,folder in (('database_path','data'),('evidence_directory','evidence')):
            value = Path(getattr(config,name))
            value = (value if value.is_absolute() else ROOT/value).resolve()
            allowed = (ROOT/folder,ROOT/'outputs')
            if not any(value==base or base in value.parents for base in allowed):
                raise ValueError('Storage paths must stay inside their configured local data roots')
            paths[name] = value
        return replace(config,**paths)


def source_identifier(source):
    if isinstance(source,int) and not isinstance(source,bool):
        return 'webcam:'+str(source)
    path = Path(source).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return 'external-file-'+hashlib.sha256(str(path).encode()).hexdigest()[:16]


class ViolationStorage:
    def __init__(self,config,source,frame_size,configuration,ocr_config=None,repository=None):
        self.config = config
        self.repository = repository or Database(config.database_path,config.busy_timeout_ms)
        self.evidence = EvidenceManager(config.evidence_directory,config.evidence_enabled,
                                        config.buffer_frames,config.annotated_evidence,config.plate_evidence)
        self.ocr_config = ocr_config or OCRConfig()
        self.session = {'session_id':uuid4().hex,'video_source':source_identifier(source),
                        'frame_size':list(frame_size),'started_at':utc_time(),
                        'configuration':dict(configuration,evidence_enabled=config.evidence_enabled,
                                             annotated_evidence=config.annotated_evidence,plate_evidence=config.plate_evidence,
                                             evidence_frame_space='resized_unannotated_processing_frame',
                                             storage={'enabled':config.enabled,'busy_timeout_ms':config.busy_timeout_ms,
                                                      'max_attempts':config.max_attempts,'retry_interval_frames':config.retry_interval_frames,
                                                      'max_pending_events':config.max_pending_events,'operations_per_frame':config.operations_per_frame,
                                                      'buffer_frames':config.buffer_frames,'retention_days':config.retention_days})}
        self.pending = OrderedDict()
        self.session_ready = False
        self.session_attempts = 0
        self.last_session_attempt = -1000000
        self.frame_index = -1
        self.operations = 0
        self.finished = False
        self.counts = Counter()
        self.database_latencies = deque(maxlen=512)
        self.evidence_latencies = deque(maxlen=512)

    def _failure(self,error):
        code = error.code if isinstance(error,StorageError) else 'invalid_data'
        self.counts['failures'] += 1
        logger.warning('Violation persistence failed (%s); candidate remains in memory',code)
        return code

    def _database_call(self,operation,*args):
        start = time.perf_counter()
        try:
            return operation(*args)
        finally:
            self.database_latencies.append(time.perf_counter()-start)

    def start(self,frame_index=-1,final=False):
        if not self.config.enabled or self.session_ready or self.session_attempts>=self.config.max_attempts:
            return self.session_ready
        if not final and frame_index-self.last_session_attempt<self.config.retry_interval_frames:
            return False
        self.session_attempts += 1
        self.last_session_attempt = frame_index
        try:
            self._database_call(self.repository.initialize)
            self._database_call(self.repository.create_session,self.session)
            self.session_ready = True
            return True
        except (StorageError,ValueError,KeyError,TypeError) as error:
            self._failure(error)
            return False

    def before_frame(self,frame_index,timestamp_ms,image,tracks):
        if not self.config.enabled:
            return
        self.frame_index = frame_index
        self.operations = 0
        self.evidence.push(frame_index,timestamp_ms,image,tracks)

    def recognition(self,event):
        meta = event.get('recognition',{})
        plate = event.get('license_plate')
        return {'status':'accepted' if plate else meta.get('status','not_attempted'),
                'license_plate':plate,'confidence':event.get('license_plate_confidence') if plate else None,
                'format_type':validate_plate(plate,self.ocr_config).format_type if plate else None,
                'observations':meta.get('observations',0)}

    def candidate(self,event,confirmation_frame):
        if not self.config.enabled:
            return
        if 'persistence' in event:
            return
        snapshot = self.evidence.snapshot(event['frame_index'])
        record = {'event_id':event['event_id'],'session_id':self.session['session_id'],
                  'vehicle_track_id':event['track_id'],'vehicle_category':None,
                  'frame_index':event['frame_index'],'confirmation_frame_index':confirmation_frame,
                  'video_timestamp_ms':event['timestamp_ms'],
                  'timestamp_kind':'derived' if event['timestamp_ms'] is not None else 'unavailable',
                  'detected_at':utc_time(),'signal_state':event['signal_state'],
                  'crossing_point':list(event['crossing_point']),
                  'vehicle_bbox':list(snapshot.boxes[event['track_id']]) if snapshot and event['track_id'] in snapshot.boxes else None,
                  'detection_status':event['status']}
        event['persistence'] = {'status':'pending','record':record,'attempts':0,
                                'last_attempt':-self.config.retry_interval_frames,'saved_recognition':None,
                                'event_saved':False,'evidence_records':{},'saved_evidence':[],'error':None}
        self._enqueue(event)
        self._persist(event,confirmation_frame)

    def _enqueue(self,event):
        if event['event_id'] in self.pending:
            return
        if len(self.pending)>=self.config.max_pending_events:
            event['persistence'].update(status='failed',error='capacity_limit')
            logger.warning('Violation persistence capacity reached; candidate remains in memory')
            return
        self.pending[event['event_id']] = event

    def _targets(self,event,final):
        crossing = event['frame_index']
        targets = {'pre':crossing-1,'crossing':crossing,'vehicle':crossing,'post':crossing+1}
        if self.config.annotated_evidence:
            targets['annotated'] = crossing
        if self.config.plate_evidence and (final or event.get('recognition',{}).get('closed') or event.get('license_plate')):
            targets['plate'] = crossing
        return targets

    def _capture_ready(self,event,frame_index,final):
        if not self.config.evidence_enabled:
            return []
        meta = event['persistence']
        for kind,target in self._targets(event,final).items():
            if kind in meta['evidence_records'] or (target>frame_index and not final):
                continue
            start = time.perf_counter()
            try:
                record = self.evidence.capture(self.session['session_id'],event,kind,target)
            finally:
                self.evidence_latencies.append(time.perf_counter()-start)
            meta['evidence_records'][kind] = record
            self.counts['evidence_'+record['capture_status']] += 1
            if record['capture_status']=='failed':
                logger.warning('Violation evidence capture failed (%s)',record['reason'])
        return [value for kind,value in meta['evidence_records'].items() if kind not in meta['saved_evidence']]

    def capture_plate(self,event,crop,frame_index,timestamp_ms):
        if not self.config.enabled or not self.config.evidence_enabled or not self.config.plate_evidence or 'persistence' not in event:
            return
        meta = event['persistence']
        if 'plate' not in meta['evidence_records'] and meta['event_saved']:
            start = time.perf_counter()
            meta['evidence_records']['plate'] = self.evidence.capture(
                self.session['session_id'],event,'plate',frame_index,image=crop,timestamp_ms=timestamp_ms)
            self.evidence_latencies.append(time.perf_counter()-start)
            self._enqueue(event)

    def _persist(self,event,frame_index,final=False):
        meta = event['persistence']
        if event['event_id'] not in self.pending or meta['attempts']>=self.config.max_attempts:
            return
        if self.operations>=self.config.operations_per_frame and not final:
            return
        if meta['error'] and not final and frame_index-meta['last_attempt']<self.config.retry_interval_frames:
            return
        recognition = self.recognition(event)
        digest = json.dumps(recognition,sort_keys=True)
        evidence_ready = self.config.evidence_enabled and (
            any(kind not in meta['saved_evidence'] for kind in meta['evidence_records']) or
            any(kind not in meta['evidence_records'] and (target<=frame_index or final)
                for kind,target in self._targets(event,final).items()))
        if meta['event_saved'] and digest==meta['saved_recognition'] and not evidence_ready:
            return  # Waiting for a future frame/crop must not consume the budget.
        if not self.start(frame_index,final):
            meta.update(status='failed' if self.session_attempts>=self.config.max_attempts else 'pending',error='session_unavailable')
            return
        self.operations += 1
        meta['attempts'] += 1
        meta['last_attempt'] = frame_index
        try:
            if not meta['event_saved']:
                inserted = self._database_call(self.repository.save_event,meta['record'],recognition)
                self.counts['events_inserted'] += bool(inserted)
                meta.update(event_saved=True,status='persisted',saved_recognition=digest)
            elif digest!=meta['saved_recognition']:
                changed = self._database_call(self.repository.update_recognition,event['event_id'],recognition)
                self.counts['recognition_updates'] += bool(changed)
                meta['saved_recognition'] = digest
            records = self._capture_ready(event,frame_index,final)
            if records:
                self._database_call(self.repository.save_evidence,event['event_id'],records)
                meta['saved_evidence'].extend(record['kind'] for record in records)
            meta.update(status='persisted',error=None,attempts=0)
            expected = {'pre','crossing','post','vehicle'} if self.config.evidence_enabled else set()
            if self.config.evidence_enabled and self.config.annotated_evidence:
                expected.add('annotated')
            if self.config.evidence_enabled and self.config.plate_evidence:
                expected.add('plate')
            if expected.issubset(meta['saved_evidence']):
                self.pending.pop(event['event_id'],None)
        except (StorageError,ValueError,KeyError,TypeError,OSError) as error:
            code = self._failure(error)
            meta.update(status='persisted' if meta['event_saved'] else 'failed' if meta['attempts']>=self.config.max_attempts else 'pending',error=code)
            if meta['attempts']>=self.config.max_attempts:
                self.pending.pop(event['event_id'],None)

    def sync(self,events,frame_index,final=False):
        if not self.config.enabled or self.finished:
            return
        self.start(frame_index,final)
        for event in events.values():
            meta = event.get('persistence')
            if meta is None or meta['attempts']>=self.config.max_attempts or meta.get('error')=='capacity_limit':
                continue
            if json.dumps(self.recognition(event),sort_keys=True)!=meta['saved_recognition']:
                self._enqueue(event)
        for event in list(self.pending.values()):
            self._persist(event,frame_index,final)

    def retry(self,event):
        """Explicit local recovery after automatic retry exhaustion; no new ID."""
        meta = event['persistence']
        meta.update(attempts=0,error=None,last_attempt=-self.config.retry_interval_frames)
        self.session_attempts = 0
        self._enqueue(event)

    def finish(self,events,status='stopped'):
        if self.finished:
            return
        if self.config.enabled:
            self.sync(events,max(self.frame_index,0),final=True)
            if self.session_ready:
                try:
                    self._database_call(self.repository.finish_session,self.session['session_id'],status)
                except (StorageError,ValueError) as error:
                    self._failure(error)
        self.finished = True
        for event in self.pending.values():
            meta = event['persistence']
            meta['automatic_retry_stopped'] = True
            if not meta['event_saved']:
                meta.update(status='failed',error=meta['error'] or 'shutdown_unpersisted')
        self.pending.clear()
        self.evidence.clear()

    def purge_expired(self,actor,dry_run=True,limit=100,now=None):
        """Local-only, explicit retention. Never called by startup or processing.

        Mark first; retain a purge-pending record on any failure. A later explicit
        run resumes removal. File/SQLite operations cannot form one atomic commit.
        """
        if not isinstance(actor,str) or not actor.strip() or not isinstance(dry_run,bool):
            raise ValueError('Local operator and explicit dry-run flag required')
        if not self.config.retention_days:
            raise ValueError('Retention is disabled; configure a positive day count')
        cutoff = datetime.fromisoformat(utc_time(now).replace('Z','+00:00'))-timedelta(days=self.config.retention_days)
        records = self.repository.history(limit=limit,before=utc_time(cutoff),session_closed=True)
        results = []
        for record in records:
            session = self.repository.get_session(record['session_id'])
            if session['ended_at'] is None:
                continue  # Active sessions cannot be removed by retention.
            results.append(record['event_id'])
            if dry_run:
                continue
            self.repository.mark_purge_pending(record['event_id'])
            for evidence in record['evidence']:
                self.evidence.remove(evidence)
            self.repository.delete_purged(record['event_id'])
        return results
