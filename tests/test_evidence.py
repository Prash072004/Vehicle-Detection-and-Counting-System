"""Synthetic, opt-in image evidence; every file lives in a temporary directory."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import cv2
import numpy as np
from evidence_manager import EvidenceManager
from database import Database
from test_database import candidate, session


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.scratch=tempfile.TemporaryDirectory(prefix='traffic-evidence-test-')
        self.addCleanup(self.scratch.cleanup)
        self.root=Path(self.scratch.name)
        self.session=session()
        self.record=candidate(self.session['session_id'])
        self.event={'event_id':self.record['event_id'],'track_id':1,'crossing_point':[30,32]}
        self.image=np.full((48,72,3),80,dtype=np.uint8)
        self.manager=EvidenceManager(self.root/'evidence',enabled=True,annotated=True)
        self.manager.push(5,150.0,self.image,[[10,2,60,45,1]])

    def capture(self,kind='crossing',frame=5):
        return self.manager.capture(self.session['session_id'],self.event,kind,frame)

    def test_disabled_default_creates_no_images_or_directory(self):
        manager=EvidenceManager(self.root/'disabled')
        manager.push(5,150,self.image,[[10,2,60,45,1]])
        self.assertIsNone(manager.frames[0].image)
        self.assertEqual(manager.capture(self.session['session_id'],self.event,'crossing',5)['capture_status'],'missing')
        self.assertFalse((self.root/'disabled').exists())

    def test_clean_frame_copy_is_independent(self):
        self.image.fill(255)
        self.assertTrue(np.all(self.manager.frames[0].image==80))

    def test_generated_relative_name_hash_and_dimensions(self):
        record=self.capture()
        self.assertEqual(record['capture_status'],'saved')
        self.assertEqual(record['relative_path'],self.session['session_id']+'/'+self.event['event_id']+'/crossing-5.jpg')
        self.assertEqual((record['width'],record['height']),(72,48))
        self.assertEqual(record['sha256'],hashlib.sha256((self.manager.root/record['relative_path']).read_bytes()).hexdigest())
        self.assertEqual(record['timestamp_kind'],'measured')

    def test_missing_pre_frame_is_explicit(self):
        result=self.capture('pre',4)
        self.assertEqual(result['capture_status'],'missing')
        self.assertIsNone(result['relative_path'])
        self.assertIsNone(result['frame_index'])

    def test_frame_zero_predecessor_is_not_fabricated(self):
        self.assertEqual(self.capture('pre',-1)['capture_status'],'missing')

    def test_vehicle_crop_and_annotation_do_not_modify_clean_frame(self):
        self.assertEqual((self.capture('vehicle')['width'],self.capture('vehicle')['height']),(50,43))
        self.assertEqual(self.capture('annotated')['capture_status'],'saved')
        self.assertTrue(np.all(self.manager.frames[0].image==80))

    def test_plate_without_crop_does_not_save_full_frame(self):
        result=self.capture('plate')
        self.assertEqual(result['reason'],'plate_crop_unavailable')
        self.assertIsNone(result['relative_path'])

    def test_explicit_plate_crop(self):
        result=self.manager.capture(self.session['session_id'],self.event,'plate',5,
                                    image=self.image[10:20,20:50],timestamp_ms=150)
        self.assertEqual((result['width'],result['height']),(30,10))

    def test_invalid_image_and_reversed_crop_rejected(self):
        for image in (np.empty((0,0,3),dtype=np.uint8),np.zeros((20,20),dtype=np.uint8),np.zeros((20,20,3),dtype=float)):
            self.assertEqual(self.manager.capture(self.session['session_id'],self.event,'plate',5,image=image)['capture_status'],'failed')
        self.assertIsNone(self.manager.crop(self.image,(30,20,10,10)))

    def test_encode_failure_reports_failed(self):
        with patch('evidence_manager.cv2.imencode',return_value=(False,None)):
            self.assertEqual(self.capture()['capture_status'],'failed')

    def test_disk_full_removes_only_own_incomplete_file(self):
        with patch('evidence_manager.os.fsync',side_effect=OSError('disk full')):
            result=self.capture()
        self.assertEqual(result['capture_status'],'failed')
        self.assertFalse(list(self.manager.root.rglob('*.jpg')))

    def test_permission_write_failure_is_explicit(self):
        with patch.object(Path,'open',side_effect=PermissionError('denied')):
            self.assertEqual(self.capture()['capture_status'],'failed')

    def test_idempotent_file_retry_does_not_overwrite(self):
        first=self.capture();path=self.manager.root/first['relative_path']
        before=path.stat().st_mtime_ns
        second=self.capture()
        self.assertEqual(first['sha256'],second['sha256'])
        self.assertEqual(path.stat().st_mtime_ns,before)

    def test_changed_existing_file_is_never_overwritten(self):
        first=self.capture();path=self.manager.root/first['relative_path']
        path.write_bytes(b'preexisting different file')
        self.assertEqual(self.capture()['reason'],'file_collision')
        self.assertEqual(path.read_bytes(),b'preexisting different file')

    def test_missing_and_changed_evidence_inspection(self):
        record=self.capture();path=self.manager.root/record['relative_path']
        self.assertEqual(self.manager.inspect(record),'saved')
        path.write_bytes(b'changed')
        self.assertEqual(self.manager.inspect(record),'changed')
        path.unlink()
        self.assertEqual(self.manager.inspect(record),'missing')

    def test_path_traversal_and_unsafe_ids_rejected(self):
        for value in ('../outside.jpg','/outside.jpg','a/../../outside.jpg','D:/outside.jpg'):
            with self.assertRaises(ValueError):
                self.manager._path(value)
        with self.assertRaises(ValueError):
            self.manager.capture('../outside',self.event,'crossing',5)

    def test_symlink_escape_rejected_where_supported(self):
        outside=self.root/'outside';outside.mkdir()
        self.manager.root.mkdir(exist_ok=True)
        link=self.manager.root/self.session['session_id']
        try:
            link.symlink_to(outside,target_is_directory=True)
        except (OSError,NotImplementedError):
            # Exercise the resolution guard even without Windows privileges.
            relative=self.session['session_id']+'/'+self.event['event_id']+'/crossing-5.jpg'
            with patch.object(Path,'resolve',return_value=outside/'escaped.jpg'),self.assertRaises(ValueError):
                self.manager._path(relative)
            return
        self.assertEqual(self.capture()['capture_status'],'failed')
        self.assertFalse(list(outside.rglob('*.jpg')))

    def test_ring_buffer_is_bounded_and_releases_frames(self):
        for frame in range(6,20):
            self.manager.push(frame,None,self.image,[])
        self.assertEqual(len(self.manager.frames),5)
        self.assertIsNone(self.manager.snapshot(5))
        self.manager.clear()
        self.assertFalse(self.manager.frames)

    def test_evidence_metadata_roundtrip_and_wrong_session_rejection(self):
        repository=Database(self.root/'test.db');repository.initialize()
        repository.create_session(self.session);repository.save_event(self.record)
        evidence=self.capture()
        repository.save_evidence(self.record['event_id'],[evidence])
        repository.save_evidence(self.record['event_id'],[evidence])
        repository.save_evidence(self.record['event_id'],[self.capture()])
        self.assertEqual(len(repository.get_event(self.record['event_id'])['evidence']),1)
        altered=dict(evidence,session_id=uuid4().hex)
        with self.assertRaises(ValueError):
            repository.save_evidence(self.record['event_id'],[altered])

    def test_cross_event_reference_cannot_inspect_or_delete_other_image(self):
        record=self.capture()
        altered=dict(record,event_id=uuid4().hex)
        with self.assertRaises(ValueError):
            self.manager.remove(altered)
        self.assertEqual(self.manager.inspect(record),'saved')


if __name__=='__main__':
    unittest.main()
