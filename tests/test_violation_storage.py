"""Bounded retries, event/OCR persistence and compatible HTTP behavior."""
from fastapi.testclient import TestClient
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from database import Database, StorageError
from violation_storage import StorageConfig, ViolationStorage, source_identifier
from test_database import candidate


def raw_event(identity=1):
    from uuid import uuid4
    return {'event_id':uuid4().hex,'track_id':identity,'frame_index':1,'timestamp_ms':10.0,
            'crossing_point':[300,320],'signal_state':'red','status':'candidate',
            'license_plate':None,'license_plate_confidence':None}


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.scratch=tempfile.TemporaryDirectory(prefix='traffic-storage-test-')
        self.addCleanup(self.scratch.cleanup)
        self.root=Path(self.scratch.name)
        self.config=replace(StorageConfig(),database_path=self.root/'test.db',evidence_directory=self.root/'evidence',busy_timeout_ms=5)
        self.service=ViolationStorage(self.config,'video/sample.mp4',(720,480),{})
        self.service.start()
        self.event=raw_event()
        self.image=np.zeros((480,720,3),dtype=np.uint8)

    def frame(self,index):
        self.service.before_frame(index,index*10.0,self.image,[[260,20,340,400,1]])

    def create(self):
        self.frame(0);self.frame(1)
        self.service.candidate(self.event,1)

    def test_disabled_storage_has_no_database(self):
        config=replace(self.config,enabled=False,database_path=self.root/'disabled.db')
        service=ViolationStorage(config,'video/sample.mp4',(720,480),{})
        service.start();service.candidate(raw_event(),1);service.finish({})
        self.assertFalse(config.database_path.exists())

    def test_candidate_persisted_without_ocr(self):
        self.create()
        row=self.service.repository.get_event(self.event['event_id'])
        self.assertIsNone(row['license_plate'])
        self.assertEqual(row['timestamp_kind'],'derived')
        self.assertEqual(row['vehicle_bbox'],[260,20,340,400])
        self.assertEqual(self.event['persistence']['status'],'persisted')
        self.assertFalse(self.config.evidence_directory.exists())

    def test_delayed_confirmation_preserves_crossing_frame(self):
        self.frame(0);self.frame(1);self.frame(2)
        self.service.candidate(self.event,2)
        row=self.service.repository.get_event(self.event['event_id'])
        self.assertEqual((row['frame_index'],row['confirmation_frame_index']),(1,2))

    def test_missing_old_bbox_is_not_replaced_with_confirmation_bbox(self):
        self.frame(10)
        self.service.candidate(self.event,10)
        self.assertIsNone(self.service.repository.get_event(self.event['event_id'])['vehicle_bbox'])

    def test_meaningful_updates_only(self):
        self.create()
        with patch.object(self.service.repository,'update_recognition',wraps=self.service.repository.update_recognition) as update:
            for index in range(2,10):
                self.frame(index);self.service.sync({1:self.event},index)
            update.assert_not_called()
            self.event['recognition']={'status':'unreadable'}
            for index in range(10,15):
                self.frame(index);self.service.sync({1:self.event},index)
            self.assertEqual(update.call_count,1)

    def test_ocr_enrichment_keeps_identity_and_crossing(self):
        self.create()
        self.event.update(license_plate='UP15AB1234',license_plate_confidence=.95,
                          recognition={'status':'accepted','observations':2})
        self.frame(2);self.service.sync({1:self.event},2)
        row=self.service.repository.get_event(self.event['event_id'])
        self.assertEqual(row['automatic_plate'],'UP15AB1234')
        self.assertEqual(row['frame_index'],1)
        self.assertEqual(row['detection_status'],'candidate')

    def test_failure_signal_change_expiry_do_not_remove_record(self):
        self.create()
        self.event['recognition']={'status':'unreadable','reason':'track_expired','closed':True}
        self.frame(2);self.service.sync({1:self.event},2)
        self.service.finish({1:self.event})
        row=Database(self.config.database_path).get_event(self.event['event_id'])
        self.assertEqual(row['recognition_status'],'unreadable')
        self.assertEqual(row['signal_state'],'red')

    def test_bounded_event_write_retries_then_explicit_recovery(self):
        with patch.object(self.service.repository,'save_event',side_effect=StorageError('locked')) as save:
            self.create()
            for index in (2,31,32,61,62,91):
                self.frame(index);self.service.sync({1:self.event},index)
            self.assertEqual(save.call_count,3)
            self.assertEqual(self.event['persistence']['status'],'failed')
        self.service.retry(self.event)
        self.frame(92);self.service.sync({1:self.event},92)
        self.assertEqual(len(self.service.repository.history()),1)

    def test_session_failure_retry_is_bounded(self):
        repository=Mock(spec=Database);repository.initialize.side_effect=StorageError('locked')
        service=ViolationStorage(self.config,'video/sample.mp4',(720,480),{},repository=repository)
        item=raw_event();service.start()
        for index in range(100):
            service.before_frame(index,None,self.image,[])
            if index==1:service.candidate(item,index)
            service.sync({1:item},index)
        self.assertEqual(repository.initialize.call_count,3)
        self.assertFalse(item['persistence']['event_saved'])
        self.assertEqual(item['persistence']['status'],'failed')

    def test_privacy_safe_failure_logs(self):
        self.event.update(license_plate='UP15AB1234',license_plate_confidence=.9)
        with patch.object(self.service.repository,'save_event',side_effect=StorageError('unavailable')), \
                self.assertLogs('violation_storage',level='WARNING') as logs:
            self.create()
        self.assertNotIn('UP15AB1234',' '.join(logs.output))
        self.assertNotIn(str(self.config.database_path),' '.join(logs.output))

    def test_evidence_opt_in_pre_crossing_post_and_crop(self):
        self.service=ViolationStorage(replace(self.config,evidence_enabled=True,annotated_evidence=True),'video/sample.mp4',(720,480),{})
        self.service.start();self.create()
        self.frame(2);self.service.sync({1:self.event},2)
        row=self.service.repository.get_event(self.event['event_id'])
        self.assertEqual({item['kind'] for item in row['evidence']},{'pre','crossing','post','vehicle','annotated'})
        self.assertTrue(all(item['capture_status']=='saved' for item in row['evidence']))

    def test_missing_post_on_shutdown_is_recorded(self):
        self.service=ViolationStorage(replace(self.config,evidence_enabled=True),'video/sample.mp4',(720,480),{})
        self.service.start();self.create()
        self.service.finish({1:self.event})
        evidence=self.service.repository.get_event(self.event['event_id'])['evidence']
        self.assertEqual(next(item for item in evidence if item['kind']=='post')['capture_status'],'missing')

    def test_evidence_db_failure_keeps_candidate_and_retries_metadata(self):
        self.service=ViolationStorage(replace(self.config,evidence_enabled=True),'video/sample.mp4',(720,480),{})
        self.service.start()
        with patch.object(self.service.repository,'save_evidence',side_effect=StorageError('locked')):
            self.create()
        self.assertIsNotNone(self.service.repository.get_event(self.event['event_id']))
        self.frame(31);self.service.sync({1:self.event},31)
        self.assertTrue(self.service.repository.get_event(self.event['event_id'])['evidence'])

    def test_operation_and_pending_limits(self):
        self.service=ViolationStorage(replace(self.config,operations_per_frame=1,max_pending_events=1),'video/sample.mp4',(720,480),{})
        self.service.start();self.frame(1)
        with patch.object(self.service.repository,'save_event',side_effect=StorageError('locked')):
            self.service.candidate(self.event,1)
            other=raw_event(2);self.service.candidate(other,1)
        self.assertEqual(len(self.service.pending),1)
        self.assertEqual(other['persistence']['error'],'capacity_limit')

    def test_finish_is_idempotent_releases_buffers(self):
        self.create()
        self.service.finish({1:self.event})
        self.service.finish({1:self.event})
        self.assertFalse(self.service.evidence.frames)
        self.assertTrue(self.service.repository.get_session(self.service.session['session_id'])['ended_at'])

    def test_retention_disabled_and_dry_run_default(self):
        with self.assertRaises(ValueError):
            self.service.purge_expired('operator')
        self.service.config=replace(self.config,retention_days=1)
        old=candidate(self.service.session['session_id']);self.service.repository.save_event(old)
        self.service.repository.finish_session(self.service.session['session_id'])
        ids=self.service.purge_expired('operator',now='2026-11-19T00:00:00Z')
        self.assertEqual(ids,[old['event_id']])
        self.assertIsNotNone(self.service.repository.get_event(old['event_id']))
        # An old but still active session must not occupy every retention page.
        from test_database import session
        active=session();self.service.repository.create_session(active)
        recent=candidate(active['session_id']);recent['detected_at']='2026-01-02T00:00:00Z'
        self.service.repository.save_event(recent)
        self.assertEqual(self.service.purge_expired('operator',limit=1,now='2026-11-19T00:00:00Z'),[old['event_id']])

    def test_explicit_retention_removes_records_and_owned_files(self):
        self.service.config=replace(self.config,retention_days=1)
        old=candidate(self.service.session['session_id']);self.service.repository.save_event(old)
        from evidence_manager import EvidenceManager
        manager=EvidenceManager(self.config.evidence_directory,enabled=True)
        manager.push(5,100,self.image,[[260,20,340,400,1]])
        record=manager.capture(self.service.session['session_id'],{'event_id':old['event_id'],'track_id':1,'crossing_point':[300,320]},'crossing',5)
        self.service.repository.save_evidence(old['event_id'],[record])
        self.service.repository.finish_session(self.service.session['session_id'])
        self.service.purge_expired('operator',dry_run=False,now='2026-11-19T00:00:00Z')
        self.assertIsNone(self.service.repository.get_event(old['event_id']))
        self.assertFalse((manager.root/record['relative_path']).exists())

    def test_configuration_validation_and_source_redaction(self):
        config=StorageConfig.from_environment(environ={'TRAFFIC_STORAGE_ENABLED':'false','TRAFFIC_RETENTION_DAYS':'30'})
        self.assertFalse(config.enabled);self.assertEqual(config.retention_days,30)
        for env in ({'TRAFFIC_EVIDENCE_ENABLED':'yes'},{'TRAFFIC_DATABASE_PATH':'../outside.db'},
                    {'TRAFFIC_EVIDENCE_DIR':'models'},{'TRAFFIC_STORAGE_MAX_ATTEMPTS':'0'}):
            with self.assertRaises(ValueError):
                StorageConfig.from_environment(environ=env)
        self.assertEqual(source_identifier(0),'webcam:0')
        self.assertNotIn('private',source_identifier(self.root/'private.mp4'))

    def test_waiting_for_plate_does_not_starve_new_events(self):
        self.service=ViolationStorage(replace(self.config,evidence_enabled=True,plate_evidence=True,operations_per_frame=1),
                                      'video/sample.mp4',(720,480),{})
        self.service.start();self.create()
        self.frame(2);self.service.sync({1:self.event},2)
        self.frame(3);self.service.sync({1:self.event},3)
        self.assertEqual(self.service.operations,0)
        other=raw_event(2);other['frame_index']=3
        self.service.candidate(other,3)
        self.assertTrue(other['persistence']['event_saved'])

    def test_retention_failure_preserves_pending_record_then_resumes(self):
        self.service.config=replace(self.config,retention_days=1)
        old=candidate(self.service.session['session_id']);self.service.repository.save_event(old)
        self.service.repository.finish_session(self.service.session['session_id'])
        with patch.object(self.service.repository,'delete_purged',side_effect=StorageError('locked')):
            with self.assertRaises(StorageError):
                self.service.purge_expired('operator',dry_run=False,now='2026-11-19T00:00:00Z')
        self.assertTrue(self.service.repository.get_event(old['event_id'])['purge_pending'])
        self.service.purge_expired('operator',dry_run=False,now='2026-11-19T00:00:00Z')
        self.assertIsNone(self.service.repository.get_event(old['event_id']))


class ApplicationStorageTests(unittest.TestCase):
    def setUp(self):
        import test_stability as stability
        self.fixture=stability.ApplicationTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.app=stability.application
        self.scratch=tempfile.TemporaryDirectory(prefix='traffic-app-storage-test-')
        self.addCleanup(self.scratch.cleanup)
        root=Path(self.scratch.name)
        self.config=replace(StorageConfig(),database_path=root/'test.db',evidence_directory=root/'evidence',busy_timeout_ms=5)
        self.addCleanup(self.app.shutdown_resources)

    def attach(self):
        service=ViolationStorage(self.config,'video/sample.mp4',(720,480),{})
        service.start()
        self.app.storage_service=service;self.app._storage_owner=self.app.crossing_detector
        return service

    def test_candidate_is_persisted_before_ocr_call(self):
        self.fixture.prepare_plate();service=self.attach()
        def recognize(*args,**kwargs):
            rows=service.repository.history()
            self.assertEqual(len(rows),1);self.assertIsNone(rows[0]['license_plate'])
            return ('AB12CDE',.9)
        self.app.read_license_plate.side_effect=recognize
        list(self.app.generate_frames())
        row=service.repository.history()[0]
        self.assertEqual(row['automatic_plate'],'AB12CDE')
        self.assertEqual(TestClient(self.app.app, follow_redirects=False).get('/violations').json(),['AB12CDE'])
        self.assertEqual(service.repository.get_session(service.session['session_id'])['processing_status'],'completed')

    def test_ocr_exception_candidate_survives_restart(self):
        self.fixture.prepare_plate();service=self.attach()
        self.app.read_license_plate.side_effect=RuntimeError('OCR failure')
        self.assertEqual(list(self.app.generate_frames()),[])
        row=Database(self.config.database_path).history()[0]
        self.assertEqual(row['recognition_status'],'unreadable')
        self.assertIsNone(row['automatic_plate'])
        self.assertEqual(service.repository.get_session(service.session['session_id'])['processing_status'],'failed')

    def test_database_failures_do_not_stop_mjpeg_or_lose_memory_event(self):
        self.fixture.prepare_plate();service=self.attach()
        with patch.object(service.repository,'save_event',side_effect=StorageError('locked')):
            self.assertEqual(len(list(self.app.generate_frames())),1)
        self.assertEqual(len(self.app.crossing_detector.events),1)
        self.assertFalse(service.repository.history())
        self.assertEqual(self.app.crossing_detector.events[7]['persistence']['status'],'failed')

    def test_history_and_review_not_exposed_as_public_routes(self):
        rules={route.path for route in self.app.app.routes}
        self.assertNotIn('/history',rules);self.assertNotIn('/review',rules)
        self.fixture.prepare_stream();service=self.attach()
        old=candidate(service.session['session_id'])
        service.repository.save_event(old,{'status':'accepted','license_plate':'UP15AB1234','confidence':.9})
        self.assertEqual(TestClient(self.app.app, follow_redirects=False).get('/violations').json(),[])


if __name__=='__main__':
    unittest.main()
