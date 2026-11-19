"""ASGI lifecycle, bounded broadcasting, transport and persistence regressions."""
import asyncio
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import test_stability as stability
from database import Database
from processing_service import ProcessingService
from violation_storage import StorageConfig, ViolationStorage


async def until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Timed out waiting for asynchronous test condition')
        await asyncio.sleep(.002)


class ProducerTests(unittest.TestCase):
    def service(self, frames, startup=None, shutdown=None, **kwargs):
        owner = ProcessingService(startup or Mock(), frames, shutdown or Mock(), **kwargs)
        self.addCleanup(owner.stop)
        return owner

    def test_constructing_service_has_no_io_or_thread(self):
        startup = Mock()
        owner = self.service(lambda: iter([]), startup)
        self.assertFalse(owner.alive)
        startup.assert_not_called()

    def test_start_initializes_once_and_waits_for_first_viewer(self):
        startup, frames = Mock(), Mock(return_value=iter([]))
        owner = self.service(frames, startup)
        owner.start(); owner.start()
        startup.assert_called_once(); frames.assert_not_called()
        self.assertTrue(owner.alive)

    def test_capture_initialization_processing_cleanup_have_same_owner_thread(self):
        ids = []
        def frames():
            ids.append(threading.get_ident())
            yield b'jpeg'
        owner = self.service(frames, lambda: ids.append(threading.get_ident()),
                             lambda: ids.append(threading.get_ident()))
        owner.start()
        async def consume():
            self.assertEqual([x async for x in owner.subscribe()], [b'jpeg'])
        asyncio.run(consume()); owner.stop()
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 1)
        self.assertNotEqual(ids[0], threading.get_ident())

    def test_two_viewers_receive_same_frames_from_one_producer(self):
        gate = threading.Event(); calls = []
        self.addCleanup(gate.set)
        def frames():
            calls.append(1)
            yield b'one'
            if not gate.wait(3):
                raise AssertionError('Viewer test gate timed out')
            yield b'two'
        owner = self.service(frames); owner.start()
        async def consume():
            first, second = owner.subscribe(), owner.subscribe()
            one = asyncio.create_task(first.__anext__())
            two = asyncio.create_task(second.__anext__())
            self.assertEqual(await one, b'one'); self.assertEqual(await two, b'one')
            self.assertEqual(owner.viewers, 2)
            gate.set()
            self.assertEqual(await first.__anext__(), b'two')
            self.assertEqual(await second.__anext__(), b'two')
            await first.aclose(); await second.aclose()
        asyncio.run(consume()); owner.stop()
        self.assertEqual(calls, [1]); self.assertEqual(owner.viewers, 0)

    def test_disconnect_one_viewer_does_not_stop_other(self):
        gate = threading.Event(); shutdown = Mock()
        self.addCleanup(gate.set)
        def frames():
            yield b'one'
            gate.wait(3)
            yield b'two'
        owner = self.service(frames, shutdown=shutdown); owner.start()
        async def consume():
            a, b = owner.subscribe(), owner.subscribe()
            self.assertEqual(await a.__anext__(), b'one')
            self.assertEqual(await b.__anext__(), b'one')
            await a.aclose()
            shutdown.assert_not_called()
            self.assertEqual(owner.viewers, 1)
            gate.set()
            self.assertEqual([x async for x in b], [b'two'])
        asyncio.run(consume()); owner.stop()
        shutdown.assert_called_once()

    def test_slow_viewer_skips_frames_and_storage_is_one_slot(self):
        gate = threading.Event(); self.addCleanup(gate.set)
        def frames():
            yield b'first'
            gate.wait(3)
            for i in range(1000):
                yield str(i).encode()
        owner = self.service(frames); owner.start()
        async def consume():
            stream = owner.subscribe()
            self.assertEqual(await stream.__anext__(), b'first')
            gate.set(); await until(lambda: owner.completed)
            self.assertEqual(await stream.__anext__(), b'999')
            with self.assertRaises(StopAsyncIteration):
                await stream.__anext__()
        asyncio.run(consume())
        self.assertEqual(owner._sequence, 1001)
        self.assertEqual(len(owner._latest), 3)

    def test_event_loop_remains_responsive_during_blocking_processing(self):
        gate = threading.Event(); entered = threading.Event()
        self.addCleanup(gate.set)
        def frames():
            entered.set(); gate.wait(3)
            yield b'done'
        owner = self.service(frames); owner.start()
        async def consume():
            stream = owner.subscribe(); task = asyncio.create_task(stream.__anext__())
            await until(entered.is_set)
            self.assertFalse(task.done())
            await asyncio.wait_for(asyncio.sleep(.01), .5)
            gate.set(); self.assertEqual(await task, b'done'); await stream.aclose()
        asyncio.run(consume())

    def test_worker_exception_closes_generator_and_resources(self):
        closed, shutdown = Mock(), Mock()
        def frames():
            try:
                yield b'one'
                raise RuntimeError('inference')
            finally:
                closed()
        owner = self.service(frames, shutdown=shutdown); owner.start()
        async def consume():
            self.assertEqual([x async for x in owner.subscribe()], [b'one'])
        with self.assertLogs('processing_service', level='ERROR'):
            asyncio.run(consume()); owner.stop()
        # A terminal inference error ends the stream; only initialization
        # failures produce the route's startup-unavailable response.
        owner.start()
        closed.assert_called_once(); shutdown.assert_called_once()

    def test_startup_failure_is_reported_and_releases_ownership(self):
        startup = Mock(side_effect=RuntimeError('bad weights')); shutdown = Mock()
        owner = self.service(lambda: iter([]), startup, shutdown)
        with self.assertLogs('processing_service', level='ERROR'), self.assertRaisesRegex(RuntimeError, 'startup failed'):
            owner.start()
        owner.stop(); shutdown.assert_called_once()
        other = self.service(lambda: iter([])); other.start()
        self.assertTrue(other.alive)

    def test_second_service_cannot_own_same_process(self):
        owner = self.service(lambda: iter([])); owner.start()
        other = self.service(lambda: iter([]))
        with self.assertRaisesRegex(RuntimeError, 'already owns'):
            other.start()

    def test_stop_before_first_viewer_is_idempotent(self):
        shutdown = Mock(); frames = Mock(return_value=iter([]))
        owner = self.service(frames, shutdown=shutdown); owner.start()
        owner.stop(); owner.stop()
        self.assertFalse(owner.alive); shutdown.assert_called_once(); frames.assert_not_called()

    def test_shutdown_timeout_does_not_release_owner_under_inference(self):
        gate = threading.Event(); entered = threading.Event()
        def frames():
            entered.set(); gate.wait(3)
            yield b'one'
        owner = self.service(frames, stop_timeout=.01); owner.start()
        owner._activate.set(); self.assertTrue(entered.wait(2))
        try:
            with self.assertRaisesRegex(RuntimeError, 'shutdown timeout'):
                owner.stop()
            other = self.service(lambda: iter([]))
            with self.assertRaisesRegex(RuntimeError, 'already owns'):
                other.start()
        finally:
            gate.set(); owner.stop_timeout=3; owner.stop()

    def test_completed_subscription_is_empty_without_restarting(self):
        calls = Mock(return_value=iter([b'one']))
        owner = self.service(calls); owner.start()
        async def consume():
            self.assertEqual([x async for x in owner.subscribe()], [b'one'])
            owner.start()
            self.assertEqual([x async for x in owner.subscribe()], [])
        asyncio.run(consume()); calls.assert_called_once()

    def test_cancelled_subscriber_releases_viewer_counter(self):
        owner = self.service(lambda: iter([])); owner.start()
        async def consume():
            # Activation is intercepted so the pending receive can be cancelled.
            with patch.object(owner._activate, 'set'):
                stream=owner.subscribe(); task=asyncio.create_task(stream.__anext__())
                await until(lambda: owner.viewers == 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(owner.viewers, 0)
        asyncio.run(consume())


class FastAPIFixture(unittest.TestCase):
    def setUp(self):
        self.fixture=stability.ApplicationTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.app=stability.application
        self.client=TestClient(self.app.app, follow_redirects=False)


class FastAPIApplicationTests(FastAPIFixture):
    def test_application_is_fastapi_and_has_no_flask_dependency(self):
        self.assertIsInstance(self.app.app, FastAPI)
        self.assertFalse(hasattr(self.app.app, 'test_client'))
        source=(stability.ROOT/'main.py').read_text()+(stability.ROOT/'web_api.py').read_text()
        self.assertNotIn('from flask', source)

    def test_template_static_paths_and_video_element(self):
        response=self.client.get('/')
        self.assertEqual(response.status_code,200)
        self.assertIn('text/html',response.headers['content-type'])
        for value in ['/static/style.css','/static/js/script.js','/video','GREEN']:
            self.assertIn(value,response.text)

    def test_css_and_javascript_served_byte_for_byte(self):
        for path in ['style.css','js/script.js']:
            response=self.client.get('/static/'+path)
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.content,(stability.ROOT/'static'/path).read_bytes())

    def test_template_static_paths_independent_of_working_directory(self):
        previous=os.getcwd()
        with tempfile.TemporaryDirectory() as scratch:
            try:
                os.chdir(scratch)
                self.assertEqual(self.client.get('/').status_code,200)
                self.assertEqual(self.client.get('/static/style.css').status_code,200)
            finally:
                os.chdir(previous)

    def test_invalid_signal_returns_validation_error_without_state_change(self):
        for color in ['yellow','RED','invalid']:
            self.assertEqual(self.client.post('/set_light/'+color).status_code,422)
        self.assertEqual(self.app.traffic_light,'green')
        self.assertEqual(self.app._signal_version,0)

    def test_signal_redirect_and_version_changes_remain_compatible(self):
        for color in ['red','red','green','red']:
            response=self.client.post('/set_light/'+color)
            self.assertEqual(response.status_code,302)
            self.assertEqual(response.headers['location'],'/')
        self.assertEqual(self.app._signal_version,3)

    def test_violations_remains_current_session_string_list(self):
        self.app.plate_memory.update({7:'AB12CDE',8:'UP15AB1234'})
        response=self.client.get('/violations')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json(),['AB12CDE','UP15AB1234'])

    def test_private_storage_review_evidence_are_not_public_routes(self):
        for path in ['/history','/review','/evidence','/data/violations.db']:
            self.assertEqual(self.client.get(path).status_code,404)

    def test_openapi_has_typed_signal_and_list_response_without_database(self):
        schema=self.client.get('/openapi.json').json()
        self.assertEqual(schema['components']['schemas']['SignalColor']['enum'],['red','green'])
        response=schema['paths']['/violations']['get']['responses']['200']['content']['application/json']['schema']
        self.assertEqual(response['type'],'array'); self.assertEqual(response['items']['type'],'string')
        self.assertEqual(set(schema['paths']),{'/','/video','/set_light/{color}','/violations'})
        self.assertEqual(self.client.get('/docs').status_code,200)

    def test_lifespan_initializes_models_once_before_first_frame(self):
        capture=stability.Capture([self.fixture.frame.copy()])
        with patch.object(self.app.cv2,'VideoCapture',return_value=capture), \
                patch.object(self.app,'_load_yolo_model',return_value=Mock()) as models, \
                patch.dict(sys.modules,self.fixture.dependencies()):
            with TestClient(self.app.app) as client:
                self.assertEqual(models.call_count,2)
                self.assertEqual(capture.read_calls,0)
                self.assertTrue(self.app.app.state.processing.alive)
                self.assertEqual(client.get('/').status_code,200)
                self.assertEqual(client.get('/violations').json(),[])
            capture.release.assert_called_once()
            self.assertFalse(self.app.app.state.processing.alive)

    def test_lifespan_startup_failure_releases_capture_and_thread(self):
        capture=stability.Capture([])
        with patch.object(self.app.cv2,'VideoCapture',return_value=capture), \
                patch.object(self.app,'_load_yolo_model',side_effect=RuntimeError('weights')), \
                self.assertLogs('processing_service',level='ERROR'), \
                self.assertRaisesRegex(RuntimeError,'startup failed'):
            with TestClient(self.app.app):
                self.fail('Failed startup must not enter lifespan')
        capture.release.assert_called_once()
        self.assertFalse(self.app.app.state.processing.alive)

    def test_application_restarts_with_new_resources_after_shutdown(self):
        captures=[]
        def prepare():
            if not self.app._initialized:
                captures.append(self.fixture.prepare_stream())
        with patch.object(self.app,'initialize_resources',side_effect=prepare):
            for _ in range(2):
                with TestClient(self.app.app) as client:
                    self.assertEqual(client.get('/video').status_code,200)
                self.assertFalse(self.app.app.state.processing.alive)
        self.assertEqual(len(captures),2)
        for capture in captures:
            capture.release.assert_called_once()

    def test_lifespan_cannot_overlap_itself(self):
        self.fixture.prepare_stream()
        async def check():
            async with self.app.app.router.lifespan_context(self.app.app):
                with self.assertRaisesRegex(RuntimeError,'already active'):
                    async with self.app.app.router.lifespan_context(self.app.app):
                        self.fail('Overlapping lifespan accepted')
        asyncio.run(check())

    def test_stream_has_original_boundary_and_valid_jpeg_and_eof(self):
        capture=self.fixture.prepare_stream()
        with TestClient(self.app.app) as client:
            response=client.get('/video')
            self.assertEqual(response.headers['content-type'],'multipart/x-mixed-replace; boundary=frame')
            self.assertTrue(response.content.startswith(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8'))
            self.assertTrue(response.content.endswith(b'\xff\xd9\r\n'))
            self.assertEqual(client.get('/video').content,b'')
            self.assertEqual(client.get('/').status_code,200)
        self.assertEqual(capture.read_calls,2); capture.release.assert_called_once()

    def test_database_session_and_ocr_enrichment_survive_fastapi_lifespan(self):
        with tempfile.TemporaryDirectory() as scratch:
            config=replace(StorageConfig(),database_path=Path(scratch)/'test.db',evidence_directory=Path(scratch)/'evidence')
            stores=[]
            def prepare():
                if self.app._initialized:
                    return
                self.fixture.prepare_plate()
                store=ViolationStorage(config,'video/sample.mp4',(720,480),{})
                store.start(); stores.append(store)
                self.app.storage_service=store; self.app._storage_owner=self.app.crossing_detector
                def recognize(*args,**kwargs):
                    rows=store.repository.history()
                    self.assertEqual(len(rows),1); self.assertIsNone(rows[0]['automatic_plate'])
                    return ('AB12CDE',.9)
                self.app.read_license_plate.side_effect=recognize
            with patch.object(self.app,'initialize_resources',side_effect=prepare):
                with TestClient(self.app.app) as client:
                    self.assertEqual(client.get('/video').status_code,200)
                    self.assertEqual(client.get('/violations').json(),['AB12CDE'])
                rows=Database(config.database_path).history()
                self.assertEqual(len(rows),1); self.assertEqual(rows[0]['automatic_plate'],'AB12CDE')
                self.assertEqual(stores[0].repository.get_session(stores[0].session['session_id'])['processing_status'],'completed')
                original_id=rows[0]['event_id']
                with TestClient(self.app.app) as client:
                    self.assertEqual(client.get('/violations').json(),[])
                    self.assertEqual(Database(config.database_path).history()[0]['event_id'],original_id)
                self.assertEqual(len(stores),2)
                self.assertNotEqual(stores[0].session['session_id'],stores[1].session['session_id'])
                self.assertEqual(stores[1].repository.get_session(stores[1].session['session_id'])['processing_status'],'stopped')

    def test_database_session_without_viewers_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as scratch:
            config=replace(StorageConfig(),database_path=Path(scratch)/'test.db',evidence_directory=Path(scratch)/'evidence')
            store=ViolationStorage(config,'video/sample.mp4',(720,480),{})
            def prepare():
                self.fixture.prepare_stream(); store.start()
                self.app.storage_service=store; self.app._storage_owner=self.app.crossing_detector
            with patch.object(self.app,'initialize_resources',side_effect=prepare):
                with TestClient(self.app.app):
                    self.assertTrue(config.database_path.exists())
                self.assertEqual(store.repository.get_session(store.session['session_id'])['processing_status'],'stopped')

    def test_import_without_flask_has_no_model_or_capture_side_effect(self):
        spec=importlib.util.spec_from_file_location('fastapi_import_check',stability.ROOT/'main.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'flask':None}), patch.object(self.app.cv2,'VideoCapture') as capture:
            spec.loader.exec_module(module)
        capture.assert_not_called(); self.assertIsNone(module.coco_model)
        self.assertFalse(module.app.state.processing.alive)


class ASGIConcurrentTests(FastAPIFixture):

    def test_multiple_subscribers_share_one_real_pipeline_session_and_event(self):
        gate=threading.Event(); self.addCleanup(gate.set)
        with tempfile.TemporaryDirectory() as scratch:
            config=replace(StorageConfig(),database_path=Path(scratch)/'test.db',
                           evidence_directory=Path(scratch)/'evidence',evidence_enabled=True)
            stores=[]; captures=[]; native_frames=self.app.generate_frames
            def prepare():
                if self.app._initialized:
                    return
                captures.append(self.fixture.prepare_plate())
                store=ViolationStorage(config,'video/sample.mp4',(720,480),{})
                store.start(); stores.append(store)
                self.app.storage_service=store; self.app._storage_owner=self.app.crossing_detector
            def frames():
                for chunk in native_frames():
                    yield chunk
                    if not gate.wait(3):
                        raise AssertionError('Persistence viewer gate timed out')
            async def check():
                async with self.app.app.router.lifespan_context(self.app.app):
                    owner=self.app.app.state.processing
                    a,b=owner.subscribe(),owner.subscribe()
                    first=asyncio.create_task(a.__anext__()); second=asyncio.create_task(b.__anext__())
                    self.assertEqual(await first,await second)
                    self.assertEqual(owner.viewers,2)
                    await a.aclose(); gate.set()
                    self.assertEqual([chunk async for chunk in b],[])
            with patch.object(self.app,'initialize_resources',side_effect=prepare), \
                    patch.object(self.app,'generate_frames',side_effect=frames):
                asyncio.run(check())
            self.assertEqual(len(stores),1)
            rows=Database(config.database_path).history()
            self.assertEqual(len(rows),1); self.assertEqual(rows[0]['automatic_plate'],'AB12CDE')
            self.app.read_license_plate.assert_called_once()
            captures[0].release.assert_called_once()
            with stores[0].repository.connection() as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM sessions').fetchone()[0],1)
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM events').fetchone()[0],1)
            for item in rows[0]['evidence']:
                if item['capture_status']=='saved':
                    self.assertEqual(stores[0].evidence.inspect(item),'saved')

    def test_simultaneous_asgi_viewers_disconnect_and_concurrent_signal_request(self):
        gate=threading.Event(); entered=threading.Event(); produced=[]
        self.addCleanup(gate.set)
        def frames():
            produced.append(1)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\none\r\n'
            entered.set()
            if not gate.wait(3):
                raise AssertionError('Concurrent viewer gate timed out')
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\ntwo\r\n'
        self.fixture.prepare_stream()
        async def check():
            disconnected=[asyncio.Event(),asyncio.Event()]; bodies=[[],[]]
            async def viewer(index):
                initial=True
                async def receive():
                    nonlocal initial
                    if initial:
                        initial=False
                        return {'type':'http.request','body':b'','more_body':False}
                    await disconnected[index].wait()
                    return {'type':'http.disconnect'}
                async def send(message):
                    if message['type']=='http.response.body' and message.get('body'):
                        bodies[index].append(message['body'])
                scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.3'},'http_version':'1.1',
                       'method':'GET','scheme':'http','path':'/video','raw_path':b'/video',
                       'root_path':'','query_string':b'','headers':[], 'server':('test',80),'client':('test',123)}
                await self.app.app(scope,receive,send)
            async with self.app.app.router.lifespan_context(self.app.app):
                tasks=[asyncio.create_task(viewer(0)),asyncio.create_task(viewer(1))]
                try:
                    # Notification delivery can now precede the producer's
                    # next generator step. Synchronize on the blocked work
                    # explicitly before testing independent request handling.
                    await until(lambda: len(bodies[0])==1 and len(bodies[1])==1 and entered.is_set())
                    self.assertTrue(entered.is_set())
                    self.assertEqual(self.app.app.state.processing.viewers,2)
                    import httpx
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app.app),base_url='http://test') as client:
                        response=await asyncio.wait_for(client.post('/set_light/red'),1)
                        self.assertEqual(response.status_code,302)
                        self.assertEqual((await client.get('/')).status_code,200)
                    disconnected[0].set(); await asyncio.wait_for(tasks[0],1)
                    self.assertEqual(self.app.app.state.processing.viewers,1)
                    gate.set(); await asyncio.wait_for(tasks[1],3)
                    self.assertEqual(len(bodies[0]),1); self.assertEqual(len(bodies[1]),2)
                    self.assertEqual(bodies[0][0],bodies[1][0])
                    self.assertEqual(produced,[1])
                finally:
                    gate.set()
                    for event in disconnected: event.set()
                    await asyncio.gather(*tasks,return_exceptions=True)
        with patch.object(self.app,'generate_frames',side_effect=frames):
            asyncio.run(check())


if __name__=='__main__':
    unittest.main()
