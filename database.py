"""Local SQLite repository. Imports open no files; each operation owns its connection.

Automatic crossing metadata is immutable. Human review is separate from detector
and OCR results. No public HTTP interface or authentication claim is provided.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path
import re
import sqlite3
from uuid import NAMESPACE_URL, uuid4, uuid5


SCHEMA_VERSION = 1
RECOGNITION_STATUSES = frozenset(('not_attempted','pending','unreadable','low_confidence',
                                 'format_unrecognized','conflicting','accepted'))
REVIEW_STATUSES = frozenset(('pending','approved','rejected'))
EVIDENCE_KINDS = frozenset(('pre','crossing','post','vehicle','plate','annotated'))
CATEGORIES = frozenset(('car','motorcycle','bus','truck'))


def utc_time(value=None):
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value,str):
        try:
            value = datetime.fromisoformat(value.replace('Z','+00:00'))
        except ValueError as error:
            raise ValueError('Timestamp must be timezone-aware ISO 8601') from error
    if not isinstance(value,datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Timestamp must be timezone aware')
    return value.astimezone(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')


def identifier(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{32}',value):
        raise ValueError('Identifier must be a generated lowercase UUID hex string')
    return value


def integer(value, name, minimum=0, maximum=2**53-1):
    if isinstance(value,bool) or not isinstance(value,int) or not minimum <= value <= maximum:
        raise ValueError(name+' must be an integer in the supported range')
    return value


def finite(value, name, minimum=0, maximum=None):
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name+' must be finite and in range')
    return float(value)


def plate_text(value, allow_empty=False):
    if not isinstance(value,str) or not re.fullmatch('[A-Z0-9]{'+('0' if allow_empty else '1')+',32}',value):
        raise ValueError('Plate must be normalized ASCII letters/digits, at most 32 characters')
    return value


class StorageError(RuntimeError):
    def __init__(self,code):
        self.code = code
        super().__init__('Violation storage operation failed ('+code+')')


def database_error(error):
    message = str(error).lower()
    if 'locked' in message or 'busy' in message:
        return StorageError('locked')
    if isinstance(error,sqlite3.IntegrityError):
        return StorageError('constraint')
    if any(word in message for word in ('schema','no such','not a database','malformed')):
        return StorageError('schema')
    return StorageError('unavailable')


SCHEMA = (
    """CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY, video_source TEXT NOT NULL,
        started_at TEXT NOT NULL, ended_at TEXT,
        processing_status TEXT NOT NULL CHECK(processing_status IN ('running','completed','stopped','failed')),
        frame_width INTEGER NOT NULL CHECK(frame_width>0), frame_height INTEGER NOT NULL CHECK(frame_height>0),
        configuration_json TEXT NOT NULL)""",
    """CREATE TABLE events (
        event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id),
        vehicle_track_id INTEGER NOT NULL CHECK(vehicle_track_id>0), vehicle_category TEXT,
        frame_index INTEGER NOT NULL CHECK(frame_index>=0), confirmation_frame_index INTEGER NOT NULL,
        video_timestamp_ms REAL, timestamp_kind TEXT NOT NULL CHECK(timestamp_kind IN ('measured','derived','unavailable')),
        detected_at TEXT NOT NULL, signal_state TEXT NOT NULL,
        crossing_x REAL NOT NULL, crossing_y REAL NOT NULL, vehicle_bbox_json TEXT,
        detection_status TEXT NOT NULL CHECK(detection_status IN ('candidate','uncertain')),
        review_status TEXT NOT NULL DEFAULT 'pending' CHECK(review_status IN ('pending','approved','rejected')),
        manual_plate TEXT, review_note TEXT, reviewed_at TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        purge_pending INTEGER NOT NULL DEFAULT 0 CHECK(purge_pending IN (0,1)),
        UNIQUE(session_id,vehicle_track_id))""",
    """CREATE TABLE plate_results (
        event_id TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
        automatic_plate TEXT, initial_accepted_plate TEXT, ocr_confidence REAL,
        format_type TEXT, recognition_status TEXT NOT NULL,
        observation_count INTEGER NOT NULL DEFAULT 0 CHECK(observation_count>=0),
        updated_at TEXT NOT NULL)""",
    """CREATE TABLE evidence (
        evidence_id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
        kind TEXT NOT NULL, relative_path TEXT, requested_frame_index INTEGER NOT NULL,
        frame_index INTEGER, video_timestamp_ms REAL, timestamp_kind TEXT NOT NULL,
        capture_status TEXT NOT NULL CHECK(capture_status IN ('saved','missing','failed')),
        reason TEXT, sha256 TEXT, width INTEGER, height INTEGER, created_at TEXT NOT NULL,
        UNIQUE(event_id,kind))""",
    """CREATE TABLE review_actions (
        action_id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
        actor TEXT NOT NULL, previous_status TEXT NOT NULL, new_status TEXT NOT NULL,
        previous_manual_plate TEXT, new_manual_plate TEXT, automatic_plate_at_review TEXT,
        note TEXT, reviewed_at TEXT NOT NULL)""",
    'CREATE INDEX events_session_time ON events(session_id,detected_at DESC,event_id DESC)',
    'CREATE INDEX events_time ON events(detected_at DESC,event_id DESC)',
    'CREATE INDEX events_review ON events(review_status,detected_at DESC,event_id DESC)',
    'CREATE INDEX plate_search ON plate_results(automatic_plate,recognition_status)',
    'CREATE INDEX manual_plate_search ON events(manual_plate)',
)
EXPECTED_COLUMNS = {
    'sessions': frozenset('session_id video_source started_at ended_at processing_status frame_width frame_height configuration_json'.split()),
    'events': frozenset('event_id session_id vehicle_track_id vehicle_category frame_index confirmation_frame_index video_timestamp_ms timestamp_kind detected_at signal_state crossing_x crossing_y vehicle_bbox_json detection_status review_status manual_plate review_note reviewed_at created_at updated_at purge_pending'.split()),
    'plate_results': frozenset('event_id automatic_plate initial_accepted_plate ocr_confidence format_type recognition_status observation_count updated_at'.split()),
    'evidence': frozenset('evidence_id event_id kind relative_path requested_frame_index frame_index video_timestamp_ms timestamp_kind capture_status reason sha256 width height created_at'.split()),
    'review_actions': frozenset('action_id event_id actor previous_status new_status previous_manual_plate new_manual_plate automatic_plate_at_review note reviewed_at'.split()),
}


class Database:
    def __init__(self,path,busy_timeout_ms=200):
        self.path = Path(path).resolve()
        if self.path.suffix.lower() not in ('.db','.sqlite','.sqlite3'):
            raise ValueError('Database requires a local .db/.sqlite/.sqlite3 path')
        self.busy_timeout_ms = integer(busy_timeout_ms,'busy_timeout_ms',0,5000)

    @contextmanager
    def connection(self,write=False,create=False):
        connection = None
        try:
            if create:
                self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            uri = self.path.as_uri()+('?mode=rwc' if create else '?mode=rw' if write else '?mode=ro')
            connection = sqlite3.connect(uri,uri=True,timeout=self.busy_timeout_ms/1000,isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA foreign_keys=ON')
            connection.execute('PRAGMA busy_timeout='+str(self.busy_timeout_ms))
            if not create and connection.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
                raise StorageError('schema')
            connection.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield connection
            connection.commit()
        except (sqlite3.Error,OSError) as error:
            if connection is not None:
                self._rollback(connection)
            raise database_error(error) from None
        except BaseException:
            if connection is not None:
                self._rollback(connection)
            raise
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _rollback(connection):
        try:
            connection.rollback()
        except sqlite3.Error:
            # Preserve the original reported failure; closing still releases the
            # failed connection. Never turn a failed operation into a success.
            pass

    def initialize(self):
        # Explicit SQL transactions avoid executescript's implicit commit behavior.
        with self.connection(write=True,create=True) as connection:
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if version == 0 and not tables:
                for statement in SCHEMA:
                    connection.execute(statement)
                connection.execute('PRAGMA user_version=1')
            elif version != SCHEMA_VERSION:
                raise StorageError('schema')
            for table,columns in EXPECTED_COLUMNS.items():
                actual = {row['name'] for row in connection.execute('PRAGMA table_info('+table+')')}
                if actual != columns:
                    raise StorageError('schema')
            normalize = lambda sql: re.sub(r'\s+',' ',sql).strip().lower()
            definitions = {normalize(row[0]) for row in connection.execute(
                "SELECT sql FROM sqlite_master WHERE type IN ('table','index') AND sql IS NOT NULL")}
            if any(normalize(statement) not in definitions for statement in SCHEMA):
                raise StorageError('schema')
            if connection.execute('PRAGMA foreign_key_check').fetchone() is not None:
                raise StorageError('schema')

    def create_session(self,session):
        identifier(session['session_id'])
        source = session['video_source']
        if not isinstance(source,str) or not source or len(source)>200 or ':' in source.replace('webcam:','') or '\\' in source or source.startswith('/') or '..' in source.split('/'):
            raise ValueError('Video source must be a safe relative identifier')
        width = integer(session['frame_size'][0],'frame_width',1,16384)
        height = integer(session['frame_size'][1],'frame_height',1,16384)
        started = utc_time(session['started_at'])
        configuration = json.dumps(session['configuration'],sort_keys=True,allow_nan=False)
        with self.connection(write=True) as connection:
            previous = connection.execute('SELECT * FROM sessions WHERE session_id=?',(session['session_id'],)).fetchone()
            values = (session['session_id'],source,started,width,height,configuration)
            if previous:
                if tuple(previous[key] for key in ('session_id','video_source','started_at','frame_width','frame_height','configuration_json')) != values:
                    raise StorageError('conflict')
                return False
            connection.execute("INSERT INTO sessions(session_id,video_source,started_at,processing_status,frame_width,frame_height,configuration_json) VALUES(?,?,?,'running',?,?,?)",values)
            return True

    def finish_session(self,session_id,status='stopped'):
        identifier(session_id)
        if status not in ('completed','stopped','failed'):
            raise ValueError('Invalid processing status')
        with self.connection(write=True) as connection:
            if connection.execute('UPDATE sessions SET processing_status=?,ended_at=? WHERE session_id=? AND ended_at IS NULL',(status,utc_time(),session_id)).rowcount:
                return True
            return False

    @staticmethod
    def validate_event(record):
        identifier(record['event_id']); identifier(record['session_id'])
        integer(record['vehicle_track_id'],'vehicle_track_id',1)
        integer(record['frame_index'],'frame_index')
        integer(record['confirmation_frame_index'],'confirmation_frame_index',record['frame_index'])
        if record['vehicle_category'] is not None and record['vehicle_category'] not in CATEGORIES:
            raise ValueError('Unsupported vehicle category')
        if record['detection_status'] not in ('candidate','uncertain') or record['signal_state'] not in ('red','green','unknown'):
            raise ValueError('Invalid detection/signal status')
        if record['detection_status']=='candidate' and record['signal_state']!='red':
            raise ValueError('Automatic candidates require a red signal')
        kind = record['timestamp_kind']
        timestamp = record['video_timestamp_ms']
        if kind not in ('measured','derived','unavailable') or ((timestamp is None)!=(kind=='unavailable')):
            raise ValueError('Timestamp value/provenance mismatch')
        if timestamp is not None:
            finite(timestamp,'video_timestamp_ms')
        point = record['crossing_point']
        if not isinstance(point,(list,tuple)) or len(point)!=2:
            raise ValueError('Crossing point requires two coordinates')
        for value in point:
            finite(value,'crossing_coordinate')
        bbox = record['vehicle_bbox']
        if bbox is not None:
            if not isinstance(bbox,(list,tuple)) or len(bbox)!=4:
                raise ValueError('Vehicle bbox requires four coordinates')
            for value in bbox:
                finite(value,'bbox_coordinate')
            if bbox[2]<=bbox[0] or bbox[3]<=bbox[1]:
                raise ValueError('Vehicle bbox must have positive area')
        return utc_time(record['detected_at'])

    def save_event(self,record,recognition=None):
        detected_at = self.validate_event(record)
        keys = ('event_id','session_id','vehicle_track_id','vehicle_category','frame_index',
                'confirmation_frame_index','video_timestamp_ms','timestamp_kind')
        values = tuple(record[key] for key in keys)+(detected_at,record['signal_state'],
                 float(record['crossing_point'][0]),float(record['crossing_point'][1]),
                 json.dumps(record['vehicle_bbox']) if record['vehicle_bbox'] is not None else None,record['detection_status'])
        columns = keys+('detected_at','signal_state','crossing_x','crossing_y','vehicle_bbox_json','detection_status')
        with self.connection(write=True) as connection:
            previous = connection.execute('SELECT * FROM events WHERE event_id=?',(record['event_id'],)).fetchone()
            if previous:
                if tuple(previous[key] for key in columns)!=values or previous['purge_pending']:
                    raise StorageError('conflict')
                inserted = False
            else:
                session = connection.execute('SELECT * FROM sessions WHERE session_id=?',(record['session_id'],)).fetchone()
                if session is None:
                    raise ValueError('Event session does not exist')
                width,height = session['frame_width'],session['frame_height']
                if not 0<=record['crossing_point'][0]<=width or not 0<=record['crossing_point'][1]<=height:
                    raise ValueError('Crossing point is outside the session frame')
                if record['vehicle_bbox'] is not None and (record['vehicle_bbox'][2]>width or record['vehicle_bbox'][3]>height):
                    raise ValueError('Vehicle bbox is outside the session frame')
                now = utc_time()
                connection.execute('INSERT INTO events('+','.join(columns)+',created_at,updated_at) VALUES('+','.join('?' for _ in range(len(values)+2))+')',values+(now,now))
                inserted = True
            if inserted or recognition is not None:
                self._recognition(connection,record['event_id'],recognition or {'status':'not_attempted'})
            return inserted

    @staticmethod
    def _recognition(connection,event_id,recognition):
        status = recognition.get('status','not_attempted')
        if status not in RECOGNITION_STATUSES:
            raise ValueError('Invalid recognition status')
        text = recognition.get('license_plate')
        score = recognition.get('confidence')
        count = integer(recognition.get('observations',0),'observation_count',0,1000000)
        kind = recognition.get('format_type')
        if kind is not None and (not isinstance(kind,str) or not kind or len(kind)>80):
            raise ValueError('Invalid format type')
        if text is not None:
            plate_text(text); finite(score,'ocr_confidence',0,1)
            if status!='accepted':
                raise ValueError('Only accepted OCR text may be persisted automatically')
        elif status=='accepted' or score is not None:
            raise ValueError('Accepted OCR requires text and confidence')
        previous = connection.execute('SELECT * FROM plate_results WHERE event_id=?',(event_id,)).fetchone()
        if previous and previous['automatic_plate'] is not None:
            if text is None or score < previous['ocr_confidence'] or (text!=previous['automatic_plate'] and score<=previous['ocr_confidence']):
                return False
            if text==previous['automatic_plate']:
                count = max(count,previous['observation_count'])
                kind = kind or previous['format_type']
        proposed = (text,score,kind,status,count)
        if previous and tuple(previous[key] for key in ('automatic_plate','ocr_confidence','format_type','recognition_status','observation_count'))==proposed:
            return False
        initial = previous['initial_accepted_plate'] if previous else None
        initial = initial or text
        now = utc_time()
        if previous:
            connection.execute('UPDATE plate_results SET automatic_plate=?,ocr_confidence=?,format_type=?,recognition_status=?,observation_count=?,initial_accepted_plate=?,updated_at=? WHERE event_id=?',proposed+(initial,now,event_id))
        else:
            connection.execute('INSERT INTO plate_results(event_id,automatic_plate,ocr_confidence,format_type,recognition_status,observation_count,initial_accepted_plate,updated_at) VALUES(?,?,?,?,?,?,?,?)',(event_id,)+proposed+(initial,now))
        connection.execute('UPDATE events SET updated_at=? WHERE event_id=?',(now,event_id))
        return True

    def update_recognition(self,event_id,recognition):
        identifier(event_id)
        with self.connection(write=True) as connection:
            row = connection.execute('SELECT purge_pending FROM events WHERE event_id=?',(event_id,)).fetchone()
            if row is None or row['purge_pending']:
                raise StorageError('conflict')
            return self._recognition(connection,event_id,recognition)

    def save_evidence(self,event_id,records):
        identifier(event_id)
        with self.connection(write=True) as connection:
            owner = connection.execute('SELECT session_id FROM events WHERE event_id=? AND purge_pending=0',(event_id,)).fetchone()
            if owner is None:
                raise StorageError('conflict')
            for record in records:
                identifier(record['evidence_id'])
                if record['event_id']!=event_id or record.get('session_id')!=owner['session_id'] or record['kind'] not in EVIDENCE_KINDS or record['capture_status'] not in ('saved','missing','failed'):
                    raise ValueError('Invalid evidence metadata')
                if record['evidence_id']!=uuid5(NAMESPACE_URL,event_id+'/'+record['kind']).hex:
                    raise ValueError('Evidence identifier does not match its event/type')
                utc_time(record['created_at'])
                integer(record['requested_frame_index'],'requested_frame_index',-1)
                saved = record['capture_status']=='saved'
                if saved:
                    integer(record['frame_index'],'frame_index')
                    relative = record['relative_path']
                    expected = identifier(record['session_id'])+'/'+event_id+'/'+record['kind']+'-'+str(record['frame_index'])+'.jpg'
                    if relative!=expected or not re.fullmatch('[0-9a-f]{64}',record['sha256'] or ''):
                        raise ValueError('Invalid evidence file reference')
                    integer(record['width'],'evidence_width',1); integer(record['height'],'evidence_height',1)
                elif any(record.get(key) is not None for key in ('relative_path','sha256','width','height')):
                    raise ValueError('Unavailable evidence cannot claim a file')
                timestamp = record.get('video_timestamp_ms')
                if timestamp is not None:
                    finite(timestamp,'evidence_timestamp')
                if record['timestamp_kind']!=('measured' if timestamp is not None else 'unavailable'):
                    raise ValueError('Invalid evidence timestamp provenance')
                columns = ('evidence_id','event_id','kind','relative_path','requested_frame_index','frame_index',
                           'video_timestamp_ms','timestamp_kind','capture_status','reason','sha256','width','height','created_at')
                values = tuple(record.get(key) for key in columns)
                previous = connection.execute('SELECT * FROM evidence WHERE evidence_id=?',(record['evidence_id'],)).fetchone()
                if previous:
                    # A retry may have a new request time; preserve the original
                    # capture creation time when all evidence facts are identical.
                    if any(previous[key]!=record.get(key) for key in columns if key!='created_at'):
                        raise StorageError('conflict')
                    continue
                connection.execute('INSERT INTO evidence('+','.join(columns)+') VALUES('+','.join('?' for _ in columns)+')',values)

    def review_event(self,event_id,new_status,actor,note=None,corrected_plate=None):
        identifier(event_id)
        if new_status not in REVIEW_STATUSES or not isinstance(actor,str) or not actor.strip() or len(actor)>80:
            raise ValueError('Valid review status and local operator identifier required')
        if note is not None and (not isinstance(note,str) or len(note)>1000):
            raise ValueError('Review note must be at most 1000 characters')
        if corrected_plate is not None:
            plate_text(corrected_plate,allow_empty=True)
        with self.connection(write=True) as connection:
            previous = connection.execute('SELECT e.*,p.automatic_plate FROM events e JOIN plate_results p USING(event_id) WHERE event_id=?',(event_id,)).fetchone()
            if previous is None or previous['purge_pending']:
                raise StorageError('conflict')
            old_status = previous['review_status']
            if old_status!=new_status and (old_status,new_status) not in (('pending','approved'),('pending','rejected'),('approved','pending'),('rejected','pending')):
                raise ValueError('Reopen a completed review before changing its decision')
            if old_status!='pending' and new_status=='pending' and not (note and note.strip()):
                raise ValueError('Reopening a review requires a note')
            manual = previous['manual_plate'] if corrected_plate is None else corrected_plate
            now = utc_time()
            connection.execute('INSERT INTO review_actions VALUES(?,?,?,?,?,?,?,?,?,?)',
                (uuid4().hex,event_id,actor,old_status,new_status,previous['manual_plate'],manual,previous['automatic_plate'],note,now))
            connection.execute('UPDATE events SET review_status=?,manual_plate=?,review_note=?,reviewed_at=?,updated_at=? WHERE event_id=?',
                               (new_status,manual,note,now,now,event_id))

    @staticmethod
    def _filters(filters):
        allowed = {'session_id','event_id','after','before','review_status','recognition_status',
                   'license_plate','vehicle_category','signal_state','purge_pending','session_closed'}
        if set(filters)-allowed:
            raise ValueError('Unknown history filter')
        clauses,arguments = [],[]
        for key,value in filters.items():
            if key in ('session_id','event_id'):
                identifier(value); column = 'e.'+key
            elif key in ('after','before'):
                value = utc_time(value)
                clauses.append('e.detected_at '+('>=' if key=='after' else '<=')+' ?')
                arguments.append(value); continue
            elif key=='review_status':
                if value not in REVIEW_STATUSES: raise ValueError('Invalid review filter')
                column = 'e.review_status'
            elif key=='recognition_status':
                if value not in RECOGNITION_STATUSES: raise ValueError('Invalid recognition filter')
                column = 'p.recognition_status'
            elif key=='license_plate':
                plate_text(value); column = 'COALESCE(e.manual_plate,p.automatic_plate)'
            elif key=='vehicle_category':
                if value not in CATEGORIES: raise ValueError('Invalid category filter')
                column = 'e.vehicle_category'
            elif key=='signal_state':
                if value not in ('red','green','unknown'): raise ValueError('Invalid signal filter')
                column = 'e.signal_state'
            elif key=='session_closed':
                if not isinstance(value,bool): raise ValueError('session_closed filter requires bool')
                clauses.append('s.ended_at IS NOT NULL' if value else 's.ended_at IS NULL')
                continue
            else:
                if not isinstance(value,bool): raise ValueError('purge_pending filter requires bool')
                value = int(value); column = 'e.purge_pending'
            clauses.append(column+' = ?'); arguments.append(value)
        if 'after' in filters and 'before' in filters and utc_time(filters['after'])>utc_time(filters['before']):
            raise ValueError('Date range is reversed')
        return (' WHERE '+' AND '.join(clauses) if clauses else ''),arguments

    @staticmethod
    def _record(connection,row):
        record = dict(row)
        record['crossing_point'] = [record.pop('crossing_x'),record.pop('crossing_y')]
        record['vehicle_bbox'] = json.loads(record.pop('vehicle_bbox_json')) if record['vehicle_bbox_json'] is not None else None
        record.pop('vehicle_bbox_json',None)
        record['license_plate'] = record['manual_plate'] if record['manual_plate'] is not None else record['automatic_plate']
        record['purge_pending'] = bool(record['purge_pending'])
        record['evidence'] = [dict(item,session_id=record['session_id']) for item in connection.execute('SELECT * FROM evidence WHERE event_id=? ORDER BY kind,evidence_id',(record['event_id'],))]
        record['review_history'] = [dict(item) for item in connection.execute('SELECT * FROM review_actions WHERE event_id=? ORDER BY reviewed_at,action_id',(record['event_id'],))]
        return record

    def history(self,limit=100,offset=0,**filters):
        integer(limit,'limit',1,500); integer(offset,'offset',0,10000000)
        where,values = self._filters(filters)
        with self.connection() as connection:
            rows = connection.execute('SELECT e.*,s.video_source,p.automatic_plate,p.initial_accepted_plate,p.ocr_confidence,p.format_type,p.recognition_status,p.observation_count FROM events e JOIN sessions s USING(session_id) JOIN plate_results p USING(event_id)'+where+' ORDER BY e.detected_at DESC,e.event_id DESC LIMIT ? OFFSET ?',tuple(values)+(limit,offset)).fetchall()
            return [self._record(connection,row) for row in rows]

    def get_event(self,event_id):
        values = self.history(event_id=identifier(event_id),limit=1)
        return values[0] if values else None

    def get_session(self,session_id):
        identifier(session_id)
        with self.connection() as connection:
            row = connection.execute('SELECT * FROM sessions WHERE session_id=?',(session_id,)).fetchone()
            if row is None:
                return None
            value = dict(row)
            value['configuration'] = json.loads(value.pop('configuration_json'))
            return value

    def mark_purge_pending(self,event_id):
        identifier(event_id)
        with self.connection(write=True) as connection:
            connection.execute('UPDATE events SET purge_pending=1 WHERE event_id=?',(event_id,))

    def delete_purged(self,event_id):
        identifier(event_id)
        with self.connection(write=True) as connection:
            if connection.execute('DELETE FROM events WHERE event_id=? AND purge_pending=1',(event_id,)).rowcount!=1:
                raise StorageError('conflict')
