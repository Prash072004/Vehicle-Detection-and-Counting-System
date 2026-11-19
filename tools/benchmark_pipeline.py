"""Sequential, isolated real-model pipeline trials. No downloads or production DB."""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import ModuleType
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from performance_monitor import PerformanceConfig, PerformanceMonitor, integer, summary

REFERENCES={'phase7':'337cffa','flask':'8626582','current':None}


def output_path(value,suffix='.json'):
    path=Path(value)
    path=(ROOT/path if not path.is_absolute() else path).resolve()
    if ROOT/'outputs' not in path.parents or path.suffix.lower()!=suffix:
        raise ValueError('Report must be a new '+suffix+' file inside outputs/')
    if path.exists(): raise ValueError('Refusing to overwrite an existing report')
    return path


def validate_device(value,torch):
    if value=='auto': return None
    if value=='cpu': return value
    if value.isascii() and value.isdigit() and torch.cuda.is_available() and int(value)<torch.cuda.device_count(): return value
    raise ValueError('Requested device unavailable; choose cpu, auto, or an available CUDA index')


def load_application(reference):
    revision=REFERENCES[reference]
    # Fixed, local trusted revisions only. No arbitrary downloaded source.
    def source(name):
        return (ROOT/name).read_text(encoding='utf-8') if revision is None else subprocess.check_output(['git','show',revision+':'+name],cwd=str(ROOT)).decode('utf-8')
    if revision and reference!='flask':
        for name in ('processing_service','web_api'):
            module=ModuleType(name);module.__file__=str(ROOT/(name+'.py'));sys.modules[name]=module
            exec(compile(source(name+'.py'),module.__file__,'exec'),module.__dict__)
    module=ModuleType('_benchmark_application');module.__file__=str(ROOT/'main.py')
    sys.modules[module.__name__]=module
    exec(compile(source('main.py'),module.__file__,'exec'),module.__dict__)
    if reference=='flask':
        from jinja2 import DictLoader
        module.app.jinja_loader=DictLoader({'index.html':source('templates/index.html')})
    return module


def equivalent(left,right):
    """Strict structural comparison; only documented subpixel float tolerance."""
    if len(left['trace'])!=len(right['trace']): return False
    import numpy as np
    for a,b in zip(left['trace'],right['trace']):
        if a['frame_index']!=b['frame_index']: return False
        for key in ('detections','tracks'):
            x,y=np.asarray(a[key]),np.asarray(b[key])
            if x.shape!=y.shape or not np.allclose(x,y,rtol=0,atol=1e-4): return False
    # UUIDs and wall-clock UTC times are deliberately excluded, not frame/time/
    # signal/point/detection status or normalized track identity.
    return left['events']==right['events'] and left['accepted_plates']==right['accepted_plates']


def run_trial(args):
    process_entry=time.perf_counter_ns()
    import asyncio
    import gc
    import logging
    import socket
    from unittest.mock import patch
    import cv2
    import numpy as np
    import psutil
    from pipeline_profiling import PipelineInstrumentation, Proxy
    from tools.detection_common import offline_environment,hardware,synchronize,sha256_file
    from violation_storage import StorageConfig,source_identifier
    for key in list(os.environ):
        if key.startswith('TRAFFIC_'): del os.environ[key]
    logging.basicConfig(level=logging.CRITICAL)
    app=load_application(args.reference)
    config=PerformanceConfig(profiling=args.profile,
                             jpeg_quality=args.jpeg_quality,pacing='unpaced')
    if args.reference=='current':app.PERFORMANCE_CONFIG=config
    monitor=PerformanceMonitor(config)
    scratch=ROOT/'outputs'/('pipeline-trial-'+uuid4().hex)
    app.VIDEO_PATH=Path(args.video).resolve()
    if not app.VIDEO_PATH.is_file(): raise ValueError('Benchmark video does not exist')
    app.STORAGE_CONFIG=replace(StorageConfig(),enabled=not args.no_storage,
        database_path=scratch/'test.db',evidence_directory=scratch/'evidence',
        evidence_enabled=args.evidence,annotated_evidence=args.evidence)
    if args.evidence and args.no_storage: raise ValueError('Evidence requires storage')
    native_cv=cv2;captures=[];models=[];cold=[];trace=[];identities={};worker_threads=set()
    acquired=[];encoded=[];encoded_bytes=[];delays=[[] for _ in range(args.viewers)]
    e2e=[[] for _ in range(args.viewers)];encode_ticks={};timer=time.perf_counter_ns
    process=psutil.Process();process.cpu_percent(None)
    memory_before=psutil.virtual_memory().available
    class Capture:
        def __init__(self,source):
            self.native=native_cv.VideoCapture(source);self.position=0;self.releases=0
            fps=self.native.get(cv2.CAP_PROP_FPS)
            self.fps=float(fps) if isinstance(fps,(int,float)) and math.isfinite(fps) and fps>0 else None
        def isOpened(self): return self.native.isOpened()
        def read(self):
            if self.position>=args.frames: return False,None
            tick=timer();ret,image=self.native.read()
            if ret and image is not None and image.size:
                acquired.append(tick);self.position+=1
            return ret,image
        def get(self,kind):
            if kind==cv2.CAP_PROP_FRAME_COUNT:
                count=self.native.get(kind);return min(count,args.frames) if count>0 else args.frames
            if kind==cv2.CAP_PROP_POS_FRAMES:return self.position
            return self.native.get(kind)
        def release(self):self.releases+=1;self.native.release()
    def capture(source):
        item=Capture(source);captures.append(item);return item
    def encode(ext,image,*params):
        # Stream-quality experiment only; evidence has its own unchanged 95.
        values=params or (() if args.jpeg_quality==95 else ([cv2.IMWRITE_JPEG_QUALITY,args.jpeg_quality],))
        result=native_cv.imencode(ext,image,*values)
        if result[0]:
            tick=timer();encoded.append(tick);data=result[1].tobytes()
            encoded_bytes.append(len(data))
        return result
    app.cv2=Proxy(native_cv,{'VideoCapture':capture,'imencode':encode})
    native_initialize=app.initialize_resources;native_load=app._load_yolo_model
    instrumentation=[None];startup_tick=[None];warmup_ns=[None];native_threads=[None]
    def load_model(path):
        tick=timer();model=native_load(path);cold.append({'model':path.name,'load_ms':(timer()-tick)/1e6})
        model.overrides.update(device=device,verbose=False);models.append(model)
        return model
    app._load_yolo_model=load_model
    def normalize_tracks(rows):
        result=[]
        for row in rows:
            identity=int(row[4]);identities.setdefault(identity,len(identities)+1)
            result.append([*map(float,row[:4]),identities[identity]])
        return result
    def initialize():
        if app._initialized:return
        startup_tick[0]=timer()
        native_initialize()
        startup_ns=timer()-startup_tick[0]
        import util
        tick=timer();reader=util.get_reader();reader_init=timer()-tick
        probe=native_cv.VideoCapture(str(app.VIDEO_PATH))
        try:ok,image=probe.read();assert ok;image=cv2.resize(image,(720,480))
        finally:probe.release()
        tick=timer()
        for _ in range(args.warmup):
            app.coco_model(image,**app.VEHICLE_DETECTION_CONFIG.inference_arguments())
            app.license_plate_detector_model(image)
        reader.readtext(np.zeros((30,100),dtype=np.uint8))
        for model in models:synchronize(model)
        if args.cpu_threads is not None:
            # Historical predictor setup also overwrites CPU budgets; explicit
            # experiments set it after warmup and verify the effective value.
            torch.set_num_threads(args.cpu_threads)
        warmup_ns[0]=timer()-tick
        cold.extend([{'startup_ms':startup_ns/1e6,'ocr_reader_init_ms':reader_init/1e6,
            'worker_entry_to_pre_warmup_ready_ms':(timer()-process_entry-warmup_ns[0])/1e6}])
        native_threads[0]=torch.get_num_threads()
        native_track=app.mot_tracker.update
        def track(rows):
            worker_threads.add(torch.get_num_threads())
            result=native_track(rows)
            trace.append({'frame_index':app._frame_index,'detections':np.asarray(rows).tolist(),
                          'tracks':normalize_tracks(result)})
            return result
        app.mot_tracker.update=track
        app.traffic_light='red'
        if args.reference=='current':
            # The benchmark controls instrumentation/pacing explicitly, keeping
            # production defaults out of a throughput experiment.
            app.PERFORMANCE_CONFIG=config
            app.performance=monitor
        instrumentation[0]=PipelineInstrumentation(app.__dict__,monitor).install()
        native_frames=app.generate_frames
        def frames():
            if args.viewers>1 and args.reference!='flask':
                deadline=time.perf_counter()+10
                while app.app.state.processing.viewers<args.viewers:
                    if time.perf_counter()>deadline:raise RuntimeError('Subscribers did not connect')
                    time.sleep(.001)
            for chunk in native_frames():
                encode_ticks[id(chunk)]=(encoded[-1],acquired[-1])
                yield chunk
        app.generate_frames=frames
        monitor.sample_resources(force=True)
        process.cpu_percent(None)
    app.initialize_resources=initialize
    if args.reference=='current':
        # Fresh child: use the same actual initializer on the owner thread,
        # with one explicitly controlled instrumentor instead of installing a
        # second set through the normal application startup wrapper.
        app._start_processing=initialize
    def deliver(chunk,index):
        if not chunk.startswith(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n') or not chunk.endswith(b'\r\n'):
            raise ValueError('Invalid MJPEG framing')
        enc,acq=encode_ticks[id(chunk)];now=timer()
        delays[index].append(now-enc);e2e[index].append(now-acq)
    # Windows event-loop socketpair must be created before denying external
    # sockets. No model/download network operation is permitted afterwards.
    loop=asyncio.new_event_loop()
    def deny(*args,**kwargs):raise RuntimeError('Network disabled during benchmark')
    try:
        with offline_environment(),patch.object(socket.socket,'connect',side_effect=deny):
            import torch
            from ultralytics import YOLO
            device=validate_device(args.device,torch)
            if args.cpu_threads is not None:torch.set_num_threads(args.cpu_threads)
            app.VEHICLE_DETECTION_CONFIG=replace(app.VEHICLE_DETECTION_CONFIG,device=device)
            if args.reference=='flask':
                if args.viewers!=1:raise ValueError('Historical Flask is compared with one viewer only')
                initialize();response=app.app.test_client().get('/video',buffered=False)
                for chunk in response.response:deliver(chunk,0)
            else:
                async def stream():
                    async with app.app.router.lifespan_context(app.app):
                        async def viewer(index):
                            first=True;never=asyncio.Event()
                            async def receive():
                                nonlocal first
                                if first:first=False;return {'type':'http.request','body':b'','more_body':False}
                                await never.wait();return {'type':'http.disconnect'}
                            async def send(message):
                                if message['type']=='http.response.start' and message['status']!=200:raise RuntimeError('Streaming failed')
                                if message['type']=='http.response.body' and message.get('body'):deliver(message['body'],index)
                            scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.3'},'http_version':'1.1',
                                   'method':'GET','scheme':'http','path':'/video','raw_path':b'/video','root_path':'',
                                   'query_string':b'','headers':[],'server':('test',80),'client':('test',123)}
                            await app.app(scope,receive,send)
                        await asyncio.gather(*(viewer(i) for i in range(args.viewers)))
                loop.run_until_complete(stream())
            app.shutdown_resources()
            if not encoded or len(encoded)!=args.frames or len(trace)!=args.frames:
                raise RuntimeError('Incomplete benchmark; no partial success report')
            if captures[0].releases!=1:raise RuntimeError('Capture was not released exactly once')
            if args.cpu_threads is not None and worker_threads!={args.cpu_threads}:
                raise RuntimeError('Requested CPU thread budget was not effective on the processing owner')
            monitor.sample_resources(force=True)
            events=sorted([{'track':identities[e['track_id']],'frame':e['frame_index'],
                'point':list(e['crossing_point']),'timestamp_ms':e['timestamp_ms'],
                'signal':e['signal_state'],'status':e['status']} for e in app.crossing_detector.events.values()],key=lambda e:e['track'])
            store=app.storage_service
            persisted=store.repository.history(session_id=store.session['session_id']) if store.config.enabled else []
            if store.config.enabled and len(persisted)!=len(events):raise RuntimeError('Candidate persistence mismatch')
            if store.config.enabled:
                originals={e['event_id']:e for e in app.crossing_detector.events.values()}
                if {row['event_id'] for row in persisted}!=set(originals):raise RuntimeError('Persisted UUID mismatch')
                for row in persisted:
                    event=originals[row['event_id']]
                    if (row['frame_index']!=event['frame_index'] or row['vehicle_track_id']!=event['track_id'] or
                            row['signal_state']!=event['signal_state'] or row['automatic_plate']!=event['license_plate']):
                        raise RuntimeError('Persisted event/enrichment mismatch')
            saved_evidence=[e for row in persisted for e in row['evidence'] if e['capture_status']=='saved']
            if any(store.evidence.inspect(e)!='saved' for e in saved_evidence):raise RuntimeError('Evidence integrity failure')
            elapsed=encoded[-1]-acquired[0];frame_ns=[end-start for start,end in zip(acquired,encoded)]
            mem=process.memory_info()
            hardware_record=hardware(models[0]);hardware_record['torch_threads_on_reporting_thread']=hardware_record['torch_threads']
            hardware_record.update(opencv_threads=cv2.getNumThreads(),torch_interop_threads=torch.get_num_interop_threads())
            hardware_record['torch_threads']=native_threads[0]
            return {'reference':args.reference,'device_requested':args.device,'hardware':hardware_record,
                'observed_processing_thread_budgets':sorted(worker_threads),
                'available_memory_before_bytes':memory_before,'available_memory_after_bytes':psutil.virtual_memory().available,
                'rss_bytes':mem.rss,'peak_rss_bytes':getattr(mem,'peak_wset',mem.rss),'process_cpu_percent':process.cpu_percent(None),
                'configuration':{'performance':asdict(config),'detection':asdict(app.VEHICLE_DETECTION_CONFIG),
                    'tracking':asdict(app.TRACKING_CONFIG),'ocr':asdict(app.OCR_CONFIG),'storage':not args.no_storage,'evidence':args.evidence,
                    'benchmark_cpu_threads':args.cpu_threads},
                'models':[{'path':'models/'+p.name,'sha256':sha256_file(p)} for p in (app.VEHICLE_MODEL_PATH,app.PLATE_MODEL_PATH)],
                'source':source_identifier(app.VIDEO_PATH),'source_sha256':sha256_file(app.VIDEO_PATH),
                'processing_size':[720,480],'frames':len(encoded),'warmup_model_calls_per_model':args.warmup,
                'source_fps':captures[0].fps,
                'model_configured_imgsz':[getattr(model.predictor.args,'imgsz',None) for model in models],
                'warmup_source_frames_consumed':0,'warmup_ms':warmup_ns[0]/1e6,'cold_start':cold,
                'ocr_warmup_calls':1,
                'processing_fps':len(encoded)/(elapsed/1e9),'processing_seconds':elapsed/1e9,
                'frame_latency':summary(frame_ns),'jpeg_bytes_mean':statistics.mean(encoded_bytes),
                'streaming_fps':[len(v)/(elapsed/1e9) for v in delays],'frames_delivered':[len(v) for v in delays],
                'subscriber_skipped':[len(encoded)-len(v) for v in delays],'detection_frames_skipped':0,
                'encode_to_asgi_send':[summary(v) for v in delays],'acquisition_to_asgi_send':[summary(v) for v in e2e],
                'candidates':len(events),'accepted_plates':len(app.plate_memory),
                'ocr_counts':dict(app.ocr_manager.counts),'persistence_counts':dict(store.counts),
                'database_relative_path':store.config.database_path.relative_to(ROOT).as_posix(),
                'evidence_relative_root':store.config.evidence_directory.relative_to(ROOT).as_posix(),
                'evidence_saved':len(saved_evidence),'capture_releases':captures[0].releases,
                'profile':monitor.snapshot(),'trace':trace,'events':events,
                'scope':'Full ordered CV pipeline, unpaced, first N frames, red signal. Steady state excludes initialization/model/OCR warmup. Delivery ends at in-process ASGI send, not browser/network. Traces contain geometry, not plate text. GPU memory only if initialized; utilization unavailable.'}
    finally:
        try:
            if args.reference!='flask':app.app.state.processing.stop()
            app.shutdown_resources()
        finally:
            if instrumentation[0]:instrumentation[0].close()
            loop.close();gc.collect()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',choices=REFERENCES,default='current')
    p.add_argument('--frames',type=int,default=60);p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--trials',type=int,default=3);p.add_argument('--viewers',type=int,default=1)
    p.add_argument('--timeout-seconds',type=int,default=300)
    p.add_argument('--video',default=str(ROOT/'video/sample.mp4'))
    p.add_argument('--device',default='cpu');p.add_argument('--cpu-threads',type=int)
    p.add_argument('--jpeg-quality',type=int,default=95)
    p.add_argument('--profile',action='store_true');p.add_argument('--evidence',action='store_true')
    p.add_argument('--no-storage',action='store_true');p.add_argument('--output',required=True)
    p.add_argument('--csv');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    try:
        integer(args.frames,'frames',1,6000);integer(args.warmup,'warmup',0,30)
        integer(args.trials,'trials',1,10);integer(args.viewers,'viewers',1,8)
        integer(args.timeout_seconds,'timeout_seconds',1,3600)
        PerformanceConfig(jpeg_quality=args.jpeg_quality)
        if args.cpu_threads is not None:integer(args.cpu_threads,'cpu_threads',1,os.cpu_count() or 1)
        path=output_path(args.output)
        csv_path=output_path(args.csv,'.csv') if args.csv else None
        if args.worker:report=run_trial(args)
        else:
            runs=[]
            forwarded=list(sys.argv[1:] if argv is None else argv)
            # Fresh subprocess per trial bounds model memory and avoids stale
            # global trackers, runtime settings and OpenMP state between trials.
            for trial in range(args.trials):
                child=ROOT/'outputs'/('pipeline-worker-'+uuid4().hex+'.json')
                command=[sys.executable,'-B',str(Path(__file__).resolve()),*forwarded,'--worker','--output',str(child)]
                try:
                    result=subprocess.run(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=args.timeout_seconds)
                except subprocess.TimeoutExpired:
                    raise RuntimeError('Benchmark worker exceeded its timeout; no partial trial accepted') from None
                if result.returncode:raise RuntimeError('Trial failed: '+result.stderr[-2000:])
                run=json.loads(child.read_text(encoding='utf-8'));runs.append(run)
                print('Trial {}: {:.3f} processing FPS; {} candidates'.format(trial+1,run['processing_fps'],run['candidates']),flush=True)
            if not all(equivalent(runs[0],r) for r in runs[1:]):raise RuntimeError('Detection/track/event behavior differs between repeated trials')
            rates=[r['processing_fps'] for r in runs]
            report={'schema_version':1,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'trials':runs,'median_processing_fps':statistics.median(rates),
                'min_processing_fps':min(rates),'max_processing_fps':max(rates),
                'mean_processing_fps':statistics.mean(rates),'equivalent_trials':True}
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as handle:json.dump(report,handle,indent=2,allow_nan=False)
        if csv_path and not args.worker:
            csv_path.parent.mkdir(parents=True,exist_ok=True)
            with csv_path.open('x',encoding='utf-8',newline='') as handle:
                writer=csv.writer(handle);writer.writerow(['trial','processing_fps','frame_mean_ms','frame_p95_ms','rss_bytes','candidates','accepted_plates'])
                for i,r in enumerate(report['trials']):writer.writerow([i+1,r['processing_fps'],r['frame_latency']['mean_ms'],r['frame_latency']['p95_ms'],r['rss_bytes'],r['candidates'],r['accepted_plates']])
        return 0
    except (ValueError,OSError,RuntimeError) as error:
        p.exit(1,'Pipeline benchmark failed: '+str(error)+'\n')


if __name__=='__main__':raise SystemExit(main())
