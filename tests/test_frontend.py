"""Dashboard integration, privacy, concurrency and frontend contract checks."""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dashboard_state import DashboardState
from test_fastapi import FastAPIFixture


def storage(enabled=True, ready=True, failures=0):
    return SimpleNamespace(config=SimpleNamespace(enabled=enabled, evidence_enabled=False),
                           session_ready=ready, counts={'failures':failures})


def event(identity, recognition='not_attempted'):
    return dict(track_id=identity, frame_index=identity, signal_state='red', status='candidate',
        recognition={'status':recognition, 'reason':'PRIVATE_REASON'},
        license_plate='PRIVATEPLATE', event_id='PRIVATE_UUID', evidence='/PRIVATE/path')


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()

    def test_uninitialized_metrics_are_unavailable(self):
        data=self.state.snapshot()
        self.assertIsNone(data['session'])
        self.assertEqual(data['metrics'], {})

    def test_ready_has_real_zero_counts_and_unknown_fps(self):
        self.state.ready(Path('/secret/source.mp4'), (720,480), 60, storage())
        data=self.state.snapshot()
        self.assertEqual(data['metrics']['processed_frames'],0)
        self.assertIsNone(data['metrics']['processing_fps'])
        self.assertNotIn('secret',json.dumps(data))
        self.assertEqual(data['source']['total_frames'],60)

    def test_unknown_source_length_is_not_invented(self):
        for value in (None,0,-1,float('nan'),float('inf')):
            self.state.ready(0,(720,480),value,None)
            self.assertIsNone(self.state.snapshot()['source']['total_frames'])
            self.assertEqual(self.state.snapshot()['source']['kind'],'camera')

    def test_counts_and_work_fps_are_computed_from_frames(self):
        self.state.frame(0,.2,3,2,{},0,30,storage())
        self.state.frame(1,.3,4,1,{7:event(7)},0,30,storage())
        metrics=self.state.snapshot()['metrics']
        self.assertEqual(metrics['processed_frames'],2)
        self.assertEqual(metrics['vehicle_detections'],7)
        self.assertEqual(metrics['observed_tracks'],1)
        self.assertEqual(metrics['candidates'],1)
        self.assertEqual(metrics['processing_fps'],4)

    def test_recent_candidates_are_bounded_redacted_and_ordered(self):
        self.state.frame(60,.1,1,1,{i:event(i) for i in range(1,61)},0,None,None)
        data=self.state.snapshot()
        self.assertEqual(len(data['recent_candidates']),20)
        self.assertEqual(data['recent_candidates'][0]['track_id'],60)
        self.assertEqual(data['metrics']['candidates'],60)
        self.assertNotIn('PRIVATE',json.dumps(data))

    def test_recognition_statuses_do_not_promote_candidates(self):
        self.state.frame(2,.1,1,1,{1:event(1,'accepted'),2:event(2,'unreadable')},1,None,None)
        rows=self.state.snapshot()['recent_candidates']
        self.assertEqual([r['recognition'] for r in rows],['unreadable','accepted'])
        self.assertTrue(all(r['status']=='candidate' for r in rows))

    def test_snapshot_does_not_reference_mutable_events(self):
        original=event(1)
        self.state.frame(1,.1,1,1,{1:original},0,None,None)
        original['recognition']['status']='accepted'
        copy=self.state.snapshot();copy['recent_candidates'].clear()
        self.assertEqual(self.state.snapshot()['recent_candidates'][0]['recognition'],'not_attempted')

    def test_reset_clears_previous_session(self):
        self.state.frame(1,.1,3,1,{1:event(1)},1,None,None)
        self.state.reset()
        self.assertEqual(self.state.snapshot()['recent_candidates'],[])
        self.assertEqual(self.state.snapshot()['metrics'],{})

    def test_terminal_outcome_survives_cleanup(self):
        for end,state in [('failed','error'),('completed','completed')]:
            self.state.reset();self.state.finish(end);self.state.finish('stopped')
            self.assertEqual(self.state.snapshot()['state'],state)

    def test_storage_disabled_failure_and_unavailable_are_distinct(self):
        for value,wanted in [(None,'unavailable'),(storage(False),'disabled'),
                             (storage(failures=1),'degraded'),(storage(ready=False),'degraded'),(storage(),'ready')]:
            self.assertEqual(self.state.storage_status(value)['state'],wanted)

    def test_final_flush_failures_are_visible_after_completion(self):
        self.state.ready(0,(720,480),None,storage())
        self.state.finish('completed',storage(failures=1))
        self.assertEqual(self.state.snapshot()['state'],'completed')
        self.assertEqual(self.state.snapshot()['storage']['state'],'degraded')

    def test_readers_receive_consistent_atomic_snapshots(self):
        def writer():
            for i in range(100):self.state.frame(i,.1,2,1,{},0,30,None)
        with ThreadPoolExecutor(max_workers=3) as pool:
            future=pool.submit(writer)
            for _ in range(100):
                data=self.state.snapshot()['metrics']
                if data:self.assertEqual(data['vehicle_detections'],data['processed_frames']*2)
            future.result()


class Markup(HTMLParser):
    def __init__(self,text):
        super().__init__();self.tags=[];self.feed(text)
    def handle_starttag(self,tag,attrs):self.tags.append((tag,dict(attrs)))


class DashboardAPITests(FastAPIFixture):
    def setUp(self):
        super().setUp()
        self.monitor=DashboardState()
        p=patch.object(self.app,'dashboard',self.monitor);p.start();self.addCleanup(p.stop)

    def test_status_does_not_initialize_models_or_database(self):
        with patch.object(self.app,'_load_yolo_model') as model,patch.object(self.app,'_storage') as store:
            response=self.client.get('/api/status')
        self.assertEqual(response.status_code,200);model.assert_not_called();store.assert_not_called()
        self.assertEqual(response.json()['state'],'not_started')

    def test_status_schema_null_metrics_and_cache_policy(self):
        response=self.client.get('/api/status');data=response.json()
        self.assertEqual(response.headers['cache-control'],'no-store')
        self.assertIsNone(data['metrics']['vehicle_detections'])
        self.assertEqual(data['access'],{'history':False,'evidence':False})
        self.assertEqual(data['schema_version'],1)

    def test_status_does_not_wait_for_inference_lock(self):
        held=Lock();held.acquire()
        try:
            with patch.object(self.app,'_resource_lock',held):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    response=pool.submit(self.client.get,'/api/status').result(timeout=2)
                self.assertEqual(response.status_code,200)
        finally:held.release()

    def test_signal_post_is_compatible_and_status_is_confirmed(self):
        response=self.client.post('/set_light/red')
        self.assertEqual(response.status_code,302)
        self.assertEqual(self.client.get('/api/status').json()['signal'],{'color':'red','version':1})
        self.client.post('/set_light/red')
        self.assertEqual(self.client.get('/api/status').json()['signal']['version'],1)

    def test_invalid_color_cannot_change_status(self):
        for color in ('yellow','RED','invalid'):
            self.assertEqual(self.client.post('/set_light/'+color).status_code,422)
        self.assertEqual(self.client.get('/api/status').json()['signal']['color'],'green')

    def test_failures_are_sanitized(self):
        self.app.app.state.processing._error=RuntimeError('PASSWORD secret D:/models/private.pt')
        response=self.client.get('/api/status');data=response.json()
        self.assertEqual(data['state'],'error')
        self.assertIn('server logs',data['error'])
        for secret in ('PASSWORD','secret','private.pt','Traceback'):
            self.assertNotIn(secret,response.text)

    def test_completed_and_failed_pipeline_outcomes_are_retained(self):
        self.monitor.finish('completed')
        self.assertEqual(self.client.get('/api/status').json()['state'],'completed')
        self.monitor.reset();self.monitor.finish('failed')
        self.assertEqual(self.client.get('/api/status').json()['state'],'error')

    def test_current_session_summary_contains_no_plate_or_evidence(self):
        self.monitor.frame(1,.1,3,2,{1:event(1,'accepted')},1,30,None)
        response=self.client.get('/api/status')
        self.assertNotIn('PRIVATE',response.text)
        self.assertEqual(response.json()['metrics']['accepted_plates'],1)
        self.app.plate_memory[1]='AB12CDE'
        self.assertEqual(self.client.get('/violations').json(),['AB12CDE'])
        self.assertEqual(self.client.get('/violations').headers['cache-control'],'no-store')

    def test_history_and_real_evidence_stay_inaccessible(self):
        for url in ('/api/history','/api/evidence','/evidence/test.jpg','/data/violations.db','/review'):
            self.assertEqual(self.client.get(url).status_code,404)

    def test_homepage_has_single_stream_semantics_and_no_fake_controls(self):
        response=self.client.get('/');markup=Markup(response.text)
        streams=[attrs for tag,attrs in markup.tags if tag=='img' and attrs.get('id')=='video-stream']
        self.assertEqual(len(streams),1);self.assertIn('/video',streams[0]['data-src'])
        self.assertEqual((streams[0]['width'],streams[0]['height']),('720','480'))
        self.assertTrue(any(tag=='main' for tag,_ in markup.tags))
        self.assertEqual(sum(tag=='h1' for tag,_ in markup.tags),1)
        self.assertIn('Skip to monitoring',response.text)
        self.assertIn('not legally confirmed violations',response.text)
        self.assertNotIn('/set_light/yellow',response.text)
        self.assertNotIn('Pause processing',response.text)

    def test_template_has_red_green_controls_and_aria_state(self):
        self.client.post('/set_light/red');html=self.client.get('/').text
        self.assertIn('>RED</strong>',html)
        forms=[a for t,a in Markup(html).tags if t=='form']
        self.assertEqual({a['action'] for a in forms},{'/set_light/red','/set_light/green'})
        self.assertIn('aria-pressed="true"',html)

    def test_synthetic_fixture_is_public_but_has_no_real_evidence(self):
        response=self.client.get('/static/evidence-fixture.svg')
        self.assertEqual(response.status_code,200)
        self.assertIn('SYNTHETIC VIEWER FIXTURE',response.text)
        self.assertIn('No vehicle, plate or event data',response.text)

    def test_responsive_and_accessible_layout_contract(self):
        css=self.client.get('/static/style.css').text
        for token in ('max-width:540px','max-width:850px','prefers-reduced-motion',
                      ':focus-visible','object-fit:contain','minmax(0,1fr)'):
            self.assertIn(token,css)
        markup=Markup(self.client.get('/').text)
        ids=[a['id'] for _,a in markup.tags if 'id' in a]
        self.assertEqual(len(ids),len(set(ids)))
        for tag,attrs in markup.tags:
            if tag=='img':self.assertTrue(attrs.get('alt'))

    def test_actual_pipeline_publishes_real_counts_without_extra_processing(self):
        capture=self.fixture.prepare_stream(detections=[[20,20,80,80,.8,2]],tracks=[])
        response=self.client.get('/video')
        self.assertEqual(response.status_code,200)
        data=self.client.get('/api/status').json()
        self.assertEqual(data['state'],'completed')
        self.assertEqual(data['metrics']['processed_frames'],1)
        self.assertEqual(data['metrics']['vehicle_detections'],1)
        self.assertEqual(data['metrics']['candidates'],0)
        self.assertGreater(data['metrics']['processing_fps'],0)
        for _ in range(20):self.assertEqual(self.client.get('/api/status').status_code,200)
        self.assertEqual(capture.read_calls,2)
        capture.release.assert_called_once()

    def test_openapi_documents_typed_status_without_sensitive_routes(self):
        schema=self.client.get('/openapi.json').json()
        model=schema['paths']['/api/status']['get']['responses']['200']['content']['application/json']['schema']
        self.assertEqual(model['$ref'],'#/components/schemas/DashboardStatus')


class JavaScriptTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'),'Node.js unavailable; run JS checks separately when available')
    def test_frontend_behavior_in_node(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run(['node','--test',str(root/'tests/frontend.test.cjs')],
                              cwd=str(root),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
