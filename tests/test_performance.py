"""Deterministic profiling/configuration/pacing checks; no FPS thresholds."""
import asyncio
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

import test_stability as stability
from performance_monitor import PerformanceConfig,PerformanceMonitor,FramePacer,STAGES,summary
from pipeline_profiling import PipelineInstrumentation
from processing_service import ProcessingService
from violation_storage import StorageConfig,ViolationStorage


class ConfigurationTests(unittest.TestCase):
    def test_defaults_preserve_encoding_and_inference(self):
        config=PerformanceConfig()
        self.assertFalse(config.profiling)
        self.assertEqual(config.jpeg_quality,95);self.assertEqual(config.pacing,'source')

    def test_invalid_boolean(self):
        for value in (1,'true',None):
            with self.subTest(value=value),self.assertRaises(ValueError):PerformanceConfig(profiling=value)

    def test_integer_bounds_reject_bool_and_fraction(self):
        for field,values in {'sample_every':[0,1001,True,1.5],'window':[0,15,4097,False],
                             'jpeg_quality':[0,101,True,90.5]}.items():
            for value in values:
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):PerformanceConfig(**{field:value})

    def test_monitor_interval_finite_and_bounded(self):
        for value in (0,float('nan'),float('inf'),61,False,'1'):
            with self.subTest(value=value),self.assertRaises(ValueError):PerformanceConfig(monitor_seconds=value)

    def test_invalid_pacing(self):
        for value in ('fast','drop','SOURCE',None):
            with self.subTest(value=value),self.assertRaises(ValueError):PerformanceConfig(pacing=value)

    def test_environment_conversion_and_no_mutation(self):
        env={'TRAFFIC_PROFILE':'true','TRAFFIC_PROFILE_EVERY':'3','TRAFFIC_PROFILE_WINDOW':'32',
             'TRAFFIC_MONITOR_SECONDS':'2','TRAFFIC_JPEG_QUALITY':'85','TRAFFIC_PACING':'unpaced'}
        default=PerformanceConfig();actual=PerformanceConfig.from_environment(default,env)
        self.assertTrue(actual.profiling);self.assertEqual(actual.sample_every,3)
        self.assertEqual(actual.jpeg_quality,85)
        self.assertFalse(default.profiling)

    def test_invalid_environment_rejected_with_name(self):
        for name,value in [('TRAFFIC_PROFILE','yes'),('TRAFFIC_PROFILE_WINDOW','x')]:
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,name):PerformanceConfig.from_environment(environ={name:value})

    def test_boolean_environment_variants(self):
        for value,expected in [('1',True),('TRUE',True),('0',False),('False',False)]:
            self.assertEqual(PerformanceConfig.from_environment(environ={'TRAFFIC_PROFILE':value}).profiling,expected)


class MonitorTests(unittest.TestCase):
    def monitor(self,**kwargs):return PerformanceMonitor(PerformanceConfig(profiling=True,**kwargs))

    def test_disabled_clock_and_wrappers_have_no_timing_work(self):
        clock=Mock(side_effect=AssertionError('Disabled profiler read clock'))
        monitor=PerformanceMonitor(clock=clock);fn=Mock(return_value=7)
        self.assertIs(monitor.wrap('capture',fn),fn)
        with monitor.span('capture'):self.assertEqual(fn(),7)
        monitor.count('captured');monitor.sample_resources()
        self.assertFalse(monitor.snapshot()['stages']);clock.assert_not_called()

    def test_nested_exclusive_accounting(self):
        monitor=self.monitor();monitor.clock=Mock(side_effect=[0,10,30,50])
        with monitor.span('pipeline'):
            with monitor.span('vehicle_inference'):pass
        stages=monitor.snapshot()['stages']
        self.assertEqual(stages['pipeline']['total_ms'],50/1e6)
        self.assertEqual(stages['pipeline']['self_total_ms'],30/1e6)
        self.assertEqual(stages['vehicle_inference']['self_total_ms'],20/1e6)

    def test_delivery_metrics_do_not_enter_processing_contributor_ranking(self):
        monitor=self.monitor();monitor.record('vehicle_inference',100)
        monitor.record('acquisition_to_subscriber',1000)
        snapshot=monitor.snapshot()
        self.assertEqual(snapshot['processing_stage_ranking'][0]['stage'],'vehicle_inference')
        self.assertIsNone(snapshot['stages']['acquisition_to_subscriber']['self_total_ms'])

    def test_exception_is_timed_and_stack_recovers(self):
        monitor=self.monitor();monitor.clock=Mock(side_effect=[0,10,20,30])
        with self.assertRaises(RuntimeError):
            with monitor.span('jpeg'):raise RuntimeError('failure')
        with monitor.span('capture'):pass
        self.assertEqual(monitor.snapshot()['stages']['capture']['self_total_ms'],10/1e6)

    def test_samples_bounded_but_lifetime_totals_preserved(self):
        monitor=self.monitor(window=16)
        for value in range(1,101):monitor.record('capture',value)
        stage=monitor.snapshot()['stages']['capture']
        self.assertEqual(stage['calls'],100);self.assertEqual(stage['samples'],16)
        self.assertEqual(stage['total_ms'],5050/1e6);self.assertEqual(stage['max_all_ms'],100/1e6)

    def test_summary_percentiles_are_not_summed(self):
        result=summary([1_000_000,2_000_000,3_000_000,4_000_000])
        self.assertEqual(result['mean_ms'],2.5);self.assertEqual(result['median_ms'],2.5)
        self.assertAlmostEqual(result['p95_ms'],3.85)
        self.assertIsNone(summary([])['mean_ms'])

    def test_labels_and_counter_names_are_whitelisted(self):
        monitor=self.monitor()
        with self.assertRaises(ValueError):monitor.record('plate-UP15AB1234',10)
        with self.assertRaises(ValueError):monitor.count('raw_vehicle_id')
        with self.assertRaises(ValueError):
            with monitor.span('path/to/private/video'):pass

    def test_invalid_measurements(self):
        monitor=self.monitor()
        for value in (-1,True,1.5):
            with self.assertRaises(ValueError):monitor.record('capture',value)
        with self.assertRaises(ValueError):monitor.record('capture',10,11)

    def test_frame_sampling_and_frame_history_bounded(self):
        monitor=self.monitor(window=16,sample_every=2);index=[-1]
        def frames():
            for i in range(50):index[0]=i;yield b'x'
        with patch.object(monitor,'sample_resources'):
            self.assertEqual(len(list(monitor.frames(frames(),lambda:index[0]))),50)
        rows=monitor.snapshot()['recent_frames']
        self.assertEqual(len(rows),16);self.assertEqual(rows[-1]['frame_index'],48)
        self.assertTrue(all(r['frame_index']%2==0 for r in rows))

    def test_frame_iterator_is_closed_on_disconnect(self):
        monitor=self.monitor();closed=Mock()
        def frames():
            try:yield b'a';yield b'b'
            finally:closed()
        with patch.object(monitor,'sample_resources'):
            iterator=monitor.frames(frames(),lambda:0);next(iterator);iterator.close()
        closed.assert_called_once()

    def test_disabled_frame_iterator_preserves_values(self):
        monitor=PerformanceMonitor()
        self.assertEqual(list(monitor.frames(iter([b'a',b'b']),lambda:0)),[b'a',b'b'])

    def test_cross_thread_stats_are_consistent(self):
        monitor=self.monitor()
        def write():
            for _ in range(50):
                with monitor.span('publication'):pass
        threads=[threading.Thread(target=write) for _ in range(3)]
        for thread in threads:thread.start()
        for thread in threads:thread.join()
        self.assertEqual(monitor.snapshot()['stages']['publication']['calls'],150)

    def test_private_snapshot_contains_no_observation_payload(self):
        monitor=self.monitor();monitor.count('ocr_calls',3);monitor.record('easyocr',1)
        data=json.dumps(monitor.snapshot(),allow_nan=False)
        self.assertNotIn('license_plate',data);self.assertNotIn('event_id',data)
        self.assertNotIn('image',data);self.assertEqual(monitor.snapshot()['counts']['ocr_calls'],3)

    def test_missing_resource_dependency_is_optional(self):
        monitor=self.monitor()
        with patch.dict('sys.modules',{'psutil':None}):monitor.sample_resources(force=True)
        self.assertEqual(monitor.snapshot()['resources'],[])


class PacingTests(unittest.TestCase):
    def test_unpaced_does_not_wait_or_read_clock(self):
        clock=Mock(side_effect=AssertionError('No timing in unpaced mode'))
        pacer=FramePacer('unpaced',30,clock=clock)
        self.assertFalse(pacer.pace(0,0));clock.assert_not_called()

    def test_source_timestamp_pacing_preserves_indices(self):
        clock=Mock(side_effect=[0,.01]);sleep=Mock()
        pacer=FramePacer('source',25,clock=clock,sleep=sleep)
        pacer.pace(0,0);pacer.pace(1,40)
        self.assertAlmostEqual(sleep.call_args[0][0],.03)

    def test_unknown_timestamps_use_source_fps(self):
        clock=Mock(side_effect=[0,.01]);sleep=Mock()
        pacer=FramePacer('source',25,clock=clock,sleep=sleep)
        pacer.pace(0,None);pacer.pace(1,None)
        self.assertAlmostEqual(sleep.call_args[0][0],.03)

    def test_slow_processing_does_not_skip_or_sleep(self):
        sleep=Mock();pacer=FramePacer('source',30,clock=Mock(side_effect=[0,2,3]),sleep=sleep)
        for i in range(3):self.assertFalse(pacer.pace(i,i*1000/30))
        sleep.assert_not_called()

    def test_shutdown_interrupts_long_source_wait(self):
        stop=Mock();stop.wait.return_value=True
        pacer=FramePacer('source',.1,stop=stop,clock=Mock(side_effect=[0,.01]))
        pacer.pace(0,0);self.assertTrue(pacer.pace(1,10000));stop.wait.assert_called_once()

    def test_unknown_invalid_fps_are_reported_unpaced(self):
        for fps in (None,0,float('nan'),float('inf'),True,1001):
            pacer=FramePacer('source',fps,sleep=Mock())
            self.assertIsNone(pacer.fps);self.assertFalse(pacer.pace(1,1000))


class InstrumentationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=stability.ApplicationTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups);self.app=stability.application
        self.config_patch=patch.object(self.app,'PERFORMANCE_CONFIG',PerformanceConfig(pacing='unpaced'))
        self.config_patch.start();self.addCleanup(self.config_patch.stop)

    def test_disabled_install_leaves_owned_objects_unchanged(self):
        self.fixture.prepare_stream();before=dict(self.app.__dict__)
        inst=PipelineInstrumentation(self.app.__dict__,PerformanceMonitor()).install()
        self.assertIs(self.app.cap,before['cap']);self.assertIs(self.app.generate_frames,before['generate_frames'])
        inst.close()

    def test_real_lifespan_installs_and_restores_optional_profiling(self):
        capture=stability.Capture([self.fixture.frame.copy()])
        model=Mock(return_value=stability.result([]))
        with patch.dict(os.environ,{'TRAFFIC_PROFILE':'true'}), \
                patch.object(self.app,'performance',PerformanceMonitor()), \
                patch.object(self.app.cv2,'VideoCapture',return_value=capture), \
                patch.object(self.app,'_load_yolo_model',return_value=model):
            with TestClient(self.app.app) as client:
                self.assertEqual(client.get('/video').status_code,200)
                state=self.app.app.state.processing.snapshot()
                self.assertTrue(state['performance']['enabled'])
                self.assertEqual(state['publication_count'],1)
                self.assertEqual(state['performance']['counts']['encoded'],1)
                self.assertEqual(state['performance']['counts']['delivered'],1)
            self.assertFalse(self.app.app.state.processing.alive)
            self.assertIsNone(self.app.cap)
        capture.release.assert_called_once()

    def test_profiled_pipeline_has_same_frame_and_single_encoding(self):
        capture=self.fixture.prepare_stream();original=self.fixture.frame.copy()
        monitor=PerformanceMonitor(PerformanceConfig(profiling=True))
        inst=PipelineInstrumentation(self.app.__dict__,monitor).install()
        try:
            with patch.object(monitor,'sample_resources'),patch.object(self.app,'performance',monitor):
                chunks=list(self.app.generate_frames())
            self.assertEqual(len(chunks),1);self.assertTrue(chunks[0].startswith(b'--frame\r\n'))
            self.assertTrue(stability.np.array_equal(self.fixture.frame,original))
            counts=monitor.snapshot()['counts']
            self.assertEqual(counts['captured'],1);self.assertEqual(counts['encoded'],1)
            self.assertNotIn('ocr_calls',counts)
        finally:inst.close()
        self.assertIsNone(self.app.cap);capture.release.assert_called_once()

    def test_profiled_ocr_retains_persistence_before_enrichment(self):
        with tempfile.TemporaryDirectory() as scratch:
            self.fixture.prepare_plate()
            config=replace(StorageConfig(),database_path=Path(scratch)/'test.db',evidence_directory=Path(scratch)/'evidence',evidence_enabled=True)
            store=ViolationStorage(config,'video/sample.mp4',(720,480),{});store.start()
            self.app.storage_service=store;self.app._storage_owner=self.app.crossing_detector
            def recognize(*args,**kwargs):
                rows=store.repository.history();self.assertEqual(len(rows),1)
                self.assertIsNone(rows[0]['automatic_plate']);return ('AB12CDE',.9)
            self.app.read_license_plate.side_effect=recognize
            monitor=PerformanceMonitor(PerformanceConfig(profiling=True))
            inst=PipelineInstrumentation(self.app.__dict__,monitor).install()
            try:
                with patch.object(monitor,'sample_resources'):self.assertEqual(len(list(self.app.generate_frames())),1)
                self.assertEqual(len(store.repository.history()),1)
                self.assertEqual(monitor.snapshot()['counts']['ocr_calls'],1)
                for kind in ('event_insert','sqlite_transaction','evidence','evidence_encode','evidence_write'):
                    self.assertIn(kind,monitor.snapshot()['stages'])
            finally:inst.close();self.app.shutdown_resources()

    def test_invalid_performance_configuration_precedes_capture(self):
        with patch.dict(os.environ,{'TRAFFIC_JPEG_QUALITY':'0'}),patch.object(self.app.cv2,'VideoCapture') as capture:
            with self.assertRaisesRegex(RuntimeError,'jpeg_quality'):self.app.initialize_resources()
        capture.assert_not_called()

    def test_stream_quality_only_changes_encoding_arguments(self):
        self.fixture.prepare_stream()
        with patch.object(self.app,'PERFORMANCE_CONFIG',PerformanceConfig(jpeg_quality=80,pacing='unpaced')), \
                patch.object(self.app.cv2,'imencode',return_value=(True,stability.np.array([1],dtype='uint8'))) as encode:
            self.assertEqual(len(list(self.app.generate_frames())),1)
        self.assertEqual(encode.call_args[0][2],[self.app.cv2.IMWRITE_JPEG_QUALITY,80])


class NotificationTests(unittest.TestCase):
    def test_joining_midstream_does_not_count_prior_frames_as_skipped(self):
        import time
        owner=ProcessingService(Mock(),lambda:iter([]),Mock())
        owner.monitor=PerformanceMonitor(PerformanceConfig(profiling=True))
        owner._sequence=10;owner._latest=(10,b'latest',(time.perf_counter_ns(),None))
        async def check():
            stream=owner.subscribe()
            self.assertEqual(await stream.__anext__(),b'latest')
            await stream.aclose()
        asyncio.run(check())
        self.assertEqual(owner.monitor.snapshot()['counts']['subscriber_skipped'],0)

    def test_stop_does_not_request_one_more_inference_frame(self):
        requested=[]
        def frames():
            requested.append(1);yield b'one'
            requested.append(2);yield b'two'
        owner=ProcessingService(Mock(),frames,Mock())
        owner._notify=owner._stop.set
        owner.start();owner._activate.set()
        owner._thread.join(3);owner.stop()
        self.assertEqual(requested,[1])

    def test_notifications_coalesce_while_loop_is_stalled_and_signal_is_pending(self):
        owner=ProcessingService(Mock(),lambda:iter([]),Mock())
        token=object();loop=Mock();event=Mock()
        owner._waiters[token]={'loop':loop,'event':event,'queued':False,'signaled':False}
        for _ in range(1000):owner._notify()
        self.assertEqual(loop.call_soon_threadsafe.call_count,1)
        owner._wake(token)
        for _ in range(1000):owner._notify()
        self.assertEqual(loop.call_soon_threadsafe.call_count,1)
        owner._waiters[token]['signaled']=False;owner._notify()
        self.assertEqual(loop.call_soon_threadsafe.call_count,2)

    def test_cancelled_subscriber_callback_is_safe(self):
        owner=ProcessingService(Mock(),lambda:iter([]),Mock())
        owner._wake(object());self.assertEqual(owner.viewers,0)

    def test_closed_event_loop_is_removed_without_worker_failure(self):
        owner=ProcessingService(Mock(),lambda:iter([]),Mock());loop=Mock()
        loop.call_soon_threadsafe.side_effect=RuntimeError('Event loop closed')
        owner._waiters[object()]={'loop':loop,'event':Mock(),'queued':False,'signaled':False}
        owner._viewers=1;owner._notify();self.assertEqual(owner.viewers,0)

    def test_service_snapshot_is_bounded_and_private(self):
        owner=ProcessingService(Mock(),lambda:iter([]),Mock());state=owner.snapshot()
        self.assertEqual(state['buffered_frames'],0);self.assertEqual(state['detection_frames_skipped'],0)
        self.assertEqual(state['processing_queue_depth'],0);self.assertNotIn('frames',state)

    def test_subscribe_uses_notification_without_polling(self):
        gate=threading.Event()
        def frames():gate.wait(3);yield b'frame'
        owner=ProcessingService(Mock(),frames,Mock());owner.start()
        async def consume():
            with patch('processing_service.asyncio.sleep',side_effect=AssertionError('Polling')):
                stream=owner.subscribe();gate.set()
                self.assertEqual([chunk async for chunk in stream],[b'frame'])
        try:asyncio.run(consume())
        finally:gate.set();owner.stop()
        self.assertEqual(owner.snapshot()['publication_count'],1)
        self.assertEqual(owner.viewers,0)


if __name__=='__main__':unittest.main()
