"""Bounded clean-frame evidence, with generated relative paths and explicit failures."""
from collections import deque
from dataclasses import dataclass
import hashlib
from contextlib import nullcontext
import os
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import cv2
import numpy as np
from database import EVIDENCE_KINDS, identifier, integer, utc_time
from plate_recognition import valid_box


@dataclass
class FrameSnapshot:
    frame_index: int
    timestamp_ms: object
    boxes: dict
    image: object = None


class EvidenceManager:
    def __init__(self,root,enabled=False,buffer_frames=5,annotated=False,plate_crops=False):
        if not all(isinstance(value,bool) for value in (enabled,annotated,plate_crops)):
            raise ValueError('Evidence flags require booleans')
        self.root = Path(root).resolve()
        self.enabled,self.annotated,self.plate_crops = enabled,annotated,plate_crops
        self.frames = deque(maxlen=integer(buffer_frames,'buffer_frames',2,30))

    def push(self,frame_index,timestamp_ms,image,tracks):
        integer(frame_index,'frame_index')
        if self.frames and frame_index<=self.frames[-1].frame_index:
            raise ValueError('Evidence frames must advance')
        boxes = {int(row[4]):tuple(float(value) for value in row[:4]) for row in tracks}
        valid_image = (isinstance(image,np.ndarray) and image.dtype==np.uint8 and
                       image.ndim==3 and image.shape[2]==3 and image.size>0)
        # Even when images are disabled, bounded geometry supplies crossing bbox
        # metadata without retaining video frames or invoking an image encoder.
        copy = image.copy() if self.enabled and valid_image else None
        self.frames.append(FrameSnapshot(frame_index,timestamp_ms,boxes,copy))

    def snapshot(self,frame_index):
        return next((frame for frame in self.frames if frame.frame_index==frame_index),None)

    def _path(self,relative):
        if not isinstance(relative,str) or not re.fullmatch(
                r'[0-9a-f]{32}/[0-9a-f]{32}/(?:pre|crossing|post|vehicle|plate|annotated)-[0-9]+\.jpg',relative):
            raise ValueError('Invalid evidence reference')
        path = self.root.joinpath(*relative.split('/'))
        resolved = path.resolve()
        if self.root not in resolved.parents or path.is_symlink():
            raise ValueError('Evidence path escapes its configured root')
        return path

    @staticmethod
    def crop(image,box):
        bounds = valid_box(box)
        if bounds is None or image is None or not image.size:
            return None
        height,width = image.shape[:2]
        x1,y1,x2,y2 = bounds
        x1,y1 = max(0,int(np.floor(x1))),max(0,int(np.floor(y1)))
        x2,y2 = min(width,int(np.ceil(x2))),min(height,int(np.ceil(y2)))
        if x2<=x1 or y2<=y1:
            return None
        return image[y1:y2,x1:x2].copy()

    def capture(self,session_id,event,kind,requested_frame_index,image=None,timestamp_ms=None):
        identifier(session_id); identifier(event['event_id'])
        if kind not in EVIDENCE_KINDS:
            raise ValueError('Unsupported evidence type')
        integer(requested_frame_index,'requested_frame_index',-1)
        evidence_id = uuid5(NAMESPACE_URL,event['event_id']+'/'+kind).hex
        record = {'evidence_id':evidence_id,'event_id':event['event_id'],'session_id':session_id,
                  'kind':kind,'relative_path':None,'requested_frame_index':requested_frame_index,
                  'frame_index':None,'video_timestamp_ms':None,'timestamp_kind':'unavailable',
                  'capture_status':'missing','reason':'frame_unavailable','sha256':None,
                  'width':None,'height':None,'created_at':utc_time()}
        if not self.enabled:
            record['reason'] = 'storage_disabled'
            return record
        if kind=='plate' and image is None:
            record['reason'] = 'plate_crop_unavailable'
            return record
        if image is None:
            snapshot = self.snapshot(requested_frame_index)
            if snapshot is None or snapshot.image is None:
                return record
            image = snapshot.image.copy()
            timestamp_ms = snapshot.timestamp_ms
            if kind=='vehicle':
                image = self.crop(image,snapshot.boxes.get(event['track_id']))
            elif kind=='annotated':
                box = snapshot.boxes.get(event['track_id'])
                if box is not None:
                    x1,y1,x2,y2 = map(int,box)
                    cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
                point = tuple(int(value) for value in event['crossing_point'])
                cv2.circle(image,point,5,(0,0,255),2)
                cv2.putText(image,'CANDIDATE - REVIEW REQUIRED',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
        if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or
                image.shape[2]!=3 or not image.size):
            record.update(capture_status='failed',reason='invalid_image_or_crop')
            return record
        record.update(frame_index=requested_frame_index,video_timestamp_ms=timestamp_ms,
                      timestamp_kind='measured' if timestamp_ms is not None else 'unavailable')
        relative = session_id+'/'+event['event_id']+'/'+kind+'-'+str(requested_frame_index)+'.jpg'
        created = False
        try:
            path = self._path(relative)
            monitor = getattr(self,'performance',None)
            with monitor.span('evidence_encode') if monitor is not None else nullcontext():
                ok,encoded = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok or encoded is None or not encoded.size:
                raise OSError('Image encoding failed')
            data = encoded.tobytes()
            digest = hashlib.sha256(data).hexdigest()
            path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            self._path(relative)  # Recheck after directory creation.
            try:
                with monitor.span('evidence_write') if monitor is not None else nullcontext():
                    with path.open('xb') as handle:
                        created = True
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
            except FileExistsError:
                # Retry can reuse an identical generated file; it cannot replace it.
                if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                    record.update(capture_status='failed',reason='file_collision')
                    return record
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise OSError('Image verification failed')
            record.update(relative_path=relative,capture_status='saved',reason=None,sha256=digest,
                          width=int(image.shape[1]),height=int(image.shape[0]))
        except (OSError,cv2.error,ValueError):
            if created:
                try:
                    self._path(relative).unlink()
                except OSError:
                    pass  # Capture remains explicitly failed; never claim an image exists.
            record.update(capture_status='failed',reason='write_failed')
        return record

    def inspect(self,record):
        if record.get('capture_status')!='saved':
            return record.get('capture_status','missing')
        path = self._record_path(record)
        try:
            if not path.is_file():
                return 'missing'
            return 'saved' if hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'] else 'changed'
        except OSError:
            return 'unavailable'

    def remove(self,record):
        if not record.get('relative_path'):
            return
        path = self._record_path(record)
        if path.exists():
            if self.inspect(record)!='saved':
                raise ValueError('Refusing to remove changed or unverified evidence')
            path.unlink()

    def _record_path(self,record):
        expected = (identifier(record['session_id'])+'/'+identifier(record['event_id'])+'/'+
                    record['kind']+'-'+str(integer(record['frame_index'],'frame_index'))+'.jpg')
        if record['relative_path']!=expected:
            raise ValueError('Evidence reference does not belong to the recorded event/frame')
        return self._path(expected)

    def clear(self):
        self.frames.clear()
