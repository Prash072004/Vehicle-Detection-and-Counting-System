"""Temporary SQLite repository tests, with no model assets or production writes."""
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from database import Database, StorageError, utc_time


def session(session_id=None):
    return {'session_id':session_id or uuid4().hex,'video_source':'video/sample.mp4',
            'started_at':'2026-01-01T00:00:00Z','frame_size':[720,480],
            'configuration':{'test':'synthetic'}}


def candidate(session_id,event_id=None,track_id=1):
    return {'event_id':event_id or uuid4().hex,'session_id':session_id,'vehicle_track_id':track_id,
            'vehicle_category':None,'frame_index':5,'confirmation_frame_index':6,
            'video_timestamp_ms':150.0,'timestamp_kind':'derived',
            'detected_at':'2026-01-01T00:00:00Z','signal_state':'red',
            'crossing_point':[300,320],'vehicle_bbox':[260,20,340,400],
            'detection_status':'candidate'}


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.scratch=tempfile.TemporaryDirectory(prefix='traffic-db-test-')
        self.addCleanup(self.scratch.cleanup)
        self.root=Path(self.scratch.name)
        self.db=Database(self.root/'nested'/'test.db',busy_timeout_ms=5)
        self.db.initialize()
        self.session=session()
        self.db.create_session(self.session)
        self.record=candidate(self.session['session_id'])

    def save(self,recognition=None):
        self.db.save_event(self.record,recognition)
        return self.db.get_event(self.record['event_id'])

    def test_import_constructor_does_not_open_database(self):
        path=self.root/'absent'/'other.db'
        Database(path)
        self.assertFalse(path.exists())

    def test_initialization_schema_and_version(self):
        with self.db.connection() as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0],1)
            tables={row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(tables,{'sessions','events','plate_results','evidence','review_actions'})

    def test_initialization_is_idempotent(self):
        self.save()
        self.db.initialize()
        self.assertIsNotNone(self.db.get_event(self.record['event_id']))

    def test_unsupported_schema_version_is_preserved(self):
        with sqlite3.connect(str(self.db.path)) as connection:
            connection.execute('PRAGMA user_version=99')
        with self.assertRaises(StorageError) as error:
            self.db.initialize()
        self.assertEqual(error.exception.code,'schema')
        with sqlite3.connect(str(self.db.path)) as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0],99)

    def test_invalid_existing_schema_not_replaced(self):
        path=self.root/'invalid.db'
        with sqlite3.connect(str(path)) as connection:
            connection.execute('CREATE TABLE unrelated(value TEXT)')
        with self.assertRaises(StorageError):
            Database(path).initialize()
        with sqlite3.connect(str(path)) as connection:
            self.assertIsNotNone(connection.execute("SELECT name FROM sqlite_master WHERE name='unrelated'").fetchone())

    def test_unavailable_database_read_does_not_create_file(self):
        path=self.root/'absent.db'
        with self.assertRaises(StorageError):
            Database(path).get_event(uuid4().hex)
        self.assertFalse(path.exists())

    def test_corrupt_file_reports_schema_without_record_content(self):
        path=self.root/'corrupt.db';path.write_bytes(b'not a database')
        with self.assertRaises(StorageError) as error:
            Database(path).initialize()
        self.assertEqual(str(error.exception),'Violation storage operation failed (schema)')

    def test_candidate_insert_and_retrieval_without_plate(self):
        record=self.save()
        self.assertIsNone(record['license_plate'])
        self.assertEqual(record['review_status'],'pending')
        self.assertEqual(record['detection_status'],'candidate')
        self.assertEqual(record['crossing_point'],[300,320])

    def test_duplicate_idempotent_insert(self):
        self.assertTrue(self.db.save_event(self.record))
        self.assertFalse(self.db.save_event(self.record))
        self.assertEqual(len(self.db.history()),1)

    def test_conflicting_same_event_cannot_change_crossing(self):
        self.save()
        other=dict(self.record,frame_index=4)
        with self.assertRaises(StorageError):
            self.db.save_event(other)
        self.assertEqual(self.db.get_event(self.record['event_id'])['frame_index'],5)

    def test_duplicate_track_same_session_different_id_is_rejected(self):
        self.save()
        with self.assertRaises(StorageError) as error:
            self.db.save_event(candidate(self.session['session_id']))
        self.assertEqual(error.exception.code,'constraint')

    def test_reused_track_in_different_session_is_distinct(self):
        self.save()
        other=session();self.db.create_session(other)
        self.db.save_event(candidate(other['session_id']))
        self.assertEqual(len(self.db.history()),2)
        self.assertEqual(len(self.db.history(session_id=self.session['session_id'])),1)

    def test_session_metadata_and_finish(self):
        self.assertFalse(self.db.create_session(self.session))
        self.assertTrue(self.db.finish_session(self.session['session_id'],'completed'))
        self.assertFalse(self.db.finish_session(self.session['session_id']))
        row=self.db.get_session(self.session['session_id'])
        self.assertEqual(row['processing_status'],'completed')
        self.assertTrue(row['ended_at'].endswith('Z'))

    def test_invalid_session_source_does_not_expose_absolute_paths(self):
        for value in ('D:\\private\\source.mp4','../outside.mp4','/secret/file'):
            other=session();other['video_source']=value
            with self.assertRaises(ValueError):
                self.db.create_session(other)

    def test_invalid_event_values_fail_before_insert(self):
        for changes in ({'crossing_point':[float('nan'),1]},{'vehicle_track_id':True},
                        {'frame_index':-1},{'vehicle_bbox':[2,2,1,1]},
                        {'signal_state':'green'},{'vehicle_category':'invented'},
                        {'timestamp_kind':'unavailable'},{'detected_at':'2026-01-01T00:00:00'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.db.save_event(dict(self.record,**changes))
        self.assertFalse(self.db.history())

    def test_out_of_frame_values_rejected(self):
        with self.assertRaises(ValueError):
            self.db.save_event(dict(self.record,crossing_point=[999,320]))

    def test_event_and_recognition_transaction_rollback(self):
        with self.assertRaises(ValueError):
            self.db.save_event(self.record,{'status':'invalid'})
        self.assertIsNone(self.db.get_event(self.record['event_id']))

    def test_explicit_transaction_rollback(self):
        self.save()
        with self.assertRaises(RuntimeError):
            with self.db.connection(write=True) as connection:
                connection.execute("UPDATE events SET review_status='approved'")
                raise RuntimeError('synthetic failure')
        self.assertEqual(self.db.get_event(self.record['event_id'])['review_status'],'pending')

    def test_disk_full_error_rolls_back_insert(self):
        with patch.object(self.db,'_recognition',side_effect=sqlite3.OperationalError('database or disk is full')):
            with self.assertRaises(StorageError):
                self.db.save_event(self.record)
        self.assertIsNone(self.db.get_event(self.record['event_id']))

    def test_new_repository_connection_survives_restart(self):
        self.save()
        self.assertEqual(Database(self.db.path).get_event(self.record['event_id'])['event_id'],self.record['event_id'])

    def test_accepted_ocr_and_weaker_result_preservation(self):
        self.save({'status':'accepted','license_plate':'UP15AB1234','confidence':.9,'format_type':'standard','observations':2})
        self.assertFalse(self.db.update_recognition(self.record['event_id'],{'status':'accepted','license_plate':'KA03MN1234','confidence':.7}))
        row=self.db.get_event(self.record['event_id'])
        self.assertEqual(row['automatic_plate'],'UP15AB1234')
        self.assertEqual(row['recognition_status'],'accepted')

    def test_stronger_ocr_keeps_initial_accepted_result(self):
        self.save({'status':'accepted','license_plate':'UP15AB1234','confidence':.8})
        self.db.update_recognition(self.record['event_id'],{'status':'accepted','license_plate':'KA03MN1234','confidence':.95})
        row=self.db.get_event(self.record['event_id'])
        self.assertEqual(row['automatic_plate'],'KA03MN1234')
        self.assertEqual(row['initial_accepted_plate'],'UP15AB1234')

    def test_unreadable_does_not_erase_accepted_result(self):
        self.save({'status':'accepted','license_plate':'UP15AB1234','confidence':.9})
        self.db.update_recognition(self.record['event_id'],{'status':'unreadable'})
        self.assertEqual(self.db.get_event(self.record['event_id'])['automatic_plate'],'UP15AB1234')

    def test_invalid_ocr_confidence_and_provisional_text_not_persisted(self):
        self.save()
        for recognition in ({'status':'accepted','license_plate':'UP15AB1234','confidence':float('nan')},
                            {'status':'pending','license_plate':'UP15AB1234','confidence':.9}):
            with self.assertRaises(ValueError):
                self.db.update_recognition(self.record['event_id'],recognition)

    def test_manual_review_preserves_automatic_and_crossing(self):
        self.save({'status':'accepted','license_plate':'UP15AB1234','confidence':.9})
        self.db.review_event(self.record['event_id'],'approved','local-reviewer',corrected_plate='KA03MN1234')
        row=self.db.get_event(self.record['event_id'])
        self.assertEqual(row['license_plate'],'KA03MN1234')
        self.assertEqual(row['automatic_plate'],'UP15AB1234')
        self.assertEqual(row['detection_status'],'candidate')
        self.assertEqual(row['crossing_point'],[300,320])
        self.assertEqual(len(row['review_history']),1)

    def test_review_reopen_requires_note_and_preserves_history(self):
        self.save()
        self.db.review_event(self.record['event_id'],'rejected','operator',note='not supported by evidence')
        with self.assertRaises(ValueError):
            self.db.review_event(self.record['event_id'],'approved','operator')
        with self.assertRaises(ValueError):
            self.db.review_event(self.record['event_id'],'pending','operator')
        self.db.review_event(self.record['event_id'],'pending','operator',note='review again')
        self.assertEqual(len(self.db.get_event(self.record['event_id'])['review_history']),2)

    def test_manual_clear_and_no_unauthorized_status_value(self):
        self.save({'status':'accepted','license_plate':'UP15AB1234','confidence':.9})
        self.db.review_event(self.record['event_id'],'pending','operator',corrected_plate='')
        self.assertEqual(self.db.get_event(self.record['event_id'])['license_plate'],'')
        with self.assertRaises(ValueError):
            self.db.review_event(self.record['event_id'],'legally_confirmed','operator')

    def test_filters_and_pagination_deterministic(self):
        self.save()
        self.db.save_event(candidate(self.session['session_id'],track_id=2))
        first=self.db.history(limit=1);second=self.db.history(limit=1,offset=1)
        self.assertNotEqual(first[0]['event_id'],second[0]['event_id'])
        self.assertEqual(self.db.history(limit=1),first)
        self.assertFalse(self.db.history(review_status='approved'))
        self.assertEqual(len(self.db.history(signal_state='red')),2)

    def test_date_range_and_empty_results(self):
        self.save()
        self.assertEqual(len(self.db.history(after='2025-12-31T00:00:00Z',before='2026-01-02T00:00:00Z')),1)
        self.assertFalse(self.db.history(after='2027-01-01T00:00:00Z'))
        with self.assertRaises(ValueError):
            self.db.history(after='2027-01-01T00:00:00Z',before='2026-01-01T00:00:00Z')

    def test_normalized_plate_and_recognition_filters(self):
        self.save({'status':'accepted','license_plate':'UP15AB1234','confidence':.9})
        self.assertEqual(len(self.db.history(license_plate='UP15AB1234',recognition_status='accepted')),1)
        self.assertFalse(self.db.history(license_plate='KA03MN1234'))

    def test_invalid_filters_and_sql_injection_rejected(self):
        self.save()
        for arguments in ({'event_id':"' OR 1=1--"},{'license_plate':"' OR 1=1--"},
                          {'order_by':'DROP TABLE events'},{'limit':1000000},{'offset':-1}):
            with self.assertRaises(ValueError):
                self.db.history(**arguments)
        self.assertEqual(len(self.db.history()),1)

    def test_database_lock_bounded_and_classified(self):
        self.save()
        with self.db.connection(write=True):
            with self.assertRaises(StorageError) as error:
                self.db.update_recognition(self.record['event_id'],{'status':'pending'})
        self.assertEqual(error.exception.code,'locked')

    def test_timestamp_timezone_and_unavailable_provenance(self):
        self.assertEqual(utc_time('2026-01-01T05:30:00+05:30'),'2026-01-01T00:00:00.000000Z')
        self.record.update(video_timestamp_ms=None,timestamp_kind='unavailable')
        self.assertEqual(self.save()['timestamp_kind'],'unavailable')

    def test_repeated_candidate_insert_cannot_reset_recognition_status(self):
        self.save({'status':'unreadable'})
        self.db.save_event(self.record)
        self.assertEqual(self.db.get_event(self.record['event_id'])['recognition_status'],'unreadable')

    def test_accepted_observation_count_does_not_regress(self):
        self.save({'status':'accepted','license_plate':'UP15AB1234','confidence':.9,'observations':3,'format_type':'standard'})
        self.db.update_recognition(self.record['event_id'],{'status':'accepted','license_plate':'UP15AB1234','confidence':.95})
        row=self.db.get_event(self.record['event_id'])
        self.assertEqual(row['observation_count'],3)
        self.assertEqual(row['format_type'],'standard')

    def test_threaded_operations_use_owned_connections(self):
        from concurrent.futures import ThreadPoolExecutor
        repository=Database(self.db.path,busy_timeout_ms=1000)
        def save(index):
            record=candidate(self.session['session_id'],track_id=index)
            repository.save_event(record)
            return repository.get_event(record['event_id'])['event_id']
        with ThreadPoolExecutor(max_workers=2) as pool:
            ids=list(pool.map(save,range(1,5)))
        self.assertEqual(len(set(ids)),4)

    def test_missing_expected_index_is_an_invalid_schema(self):
        with self.db.connection(write=True) as connection:
            connection.execute('DROP INDEX events_review')
        with self.assertRaises(StorageError) as error:
            self.db.initialize()
        self.assertEqual(error.exception.code,'schema')

    def test_abrupt_process_exit_rolls_back_uncommitted_transaction(self):
        import subprocess,sys
        self.save()
        script=('import os,sys; from database import Database; db=Database(sys.argv[1]); '
                'ctx=db.connection(write=True); connection=ctx.__enter__(); '
                'connection.execute("UPDATE events SET review_status=\'approved\'"); os._exit(23)')
        result=subprocess.run([sys.executable,'-B','-c',script,str(self.db.path)],cwd=str(Path(__file__).resolve().parents[1]),capture_output=True)
        self.assertEqual(result.returncode,23)
        self.db.initialize()
        self.assertEqual(self.db.get_event(self.record['event_id'])['review_status'],'pending')


if __name__=='__main__':
    unittest.main()
