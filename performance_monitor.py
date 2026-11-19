"""Private, bounded timings. No images, plates, event IDs or public endpoints."""
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
import os
from threading import Lock, local
import time


STAGES = frozenset(('pipeline','capture','resize','frame_preparation','vehicle_detection',
    'vehicle_inference','tracking_lifecycle','sort','crossing','plate_inference',
    'association','crop','preprocess','ocr_pipeline','easyocr','consensus','storage',
    'sqlite','sqlite_transaction','event_insert','ocr_update','evidence_metadata',
    'evidence','evidence_encode','evidence_write','annotation','jpeg','publication',
    'subscriber_delay','acquisition_to_subscriber','pacing','lock_wait'))


def integer(value, name, low, high):
    if isinstance(value,bool) or not isinstance(value,int) or not low <= value <= high:
        raise ValueError('{} must be an integer in [{}, {}]'.format(name,low,high))
    return value


@dataclass(frozen=True)
class PerformanceConfig:
    profiling: bool = False
    sample_every: int = 1
    window: int = 512
    monitor_seconds: float = 1.0
    jpeg_quality: int = 95
    pacing: str = 'source'

    def __post_init__(self):
        if not isinstance(self.profiling,bool): raise ValueError('profiling must be boolean')
        integer(self.sample_every,'sample_every',1,1000)
        integer(self.window,'window',16,4096)
        integer(self.jpeg_quality,'jpeg_quality',1,100)
        if self.pacing not in ('source','unpaced'): raise ValueError('pacing must be source or unpaced')
        if isinstance(self.monitor_seconds,bool) or not isinstance(self.monitor_seconds,(int,float)) or not math.isfinite(self.monitor_seconds) or not .1 <= self.monitor_seconds <= 60:
            raise ValueError('monitor_seconds must be finite and in [0.1, 60]')

    @classmethod
    def from_environment(cls,default=None,environ=None):
        env=os.environ if environ is None else environ
        changes={}
        for variable,field,convert in (
            ('TRAFFIC_PROFILE','profiling',str),('TRAFFIC_PROFILE_EVERY','sample_every',int),
            ('TRAFFIC_PROFILE_WINDOW','window',int),('TRAFFIC_MONITOR_SECONDS','monitor_seconds',float),
            ('TRAFFIC_JPEG_QUALITY','jpeg_quality',int),('TRAFFIC_PACING','pacing',str)):
            if variable in env:
                try: changes[field]=convert(env[variable])
                except (ValueError,TypeError): raise ValueError('Invalid '+variable) from None
        if 'profiling' in changes:
            if changes['profiling'].lower() not in ('true','false','1','0'):
                raise ValueError('TRAFFIC_PROFILE must be true, false, 1 or 0')
            changes['profiling']=changes['profiling'].lower() in ('true','1')
        return replace(default or cls(),**changes)


def summary(values):
    values=sorted(values)
    if not values: return {'samples':0,'mean_ms':None,'median_ms':None,'p95_ms':None,'max_ms':None}
    def quantile(fraction):
        position=(len(values)-1)*fraction; low=int(position); high=min(low+1,len(values)-1)
        return (values[low]+(values[high]-values[low])*(position-low))/1e6
    return {'samples':len(values),'mean_ms':sum(values)/len(values)/1e6,
            'median_ms':quantile(.5),'p95_ms':quantile(.95),'max_ms':values[-1]/1e6}


class PerformanceMonitor:
    def __init__(self,config=None,clock=time.perf_counter_ns):
        self.config=config or PerformanceConfig()
        self.enabled=self.config.profiling
        self.clock=clock
        self._local=local(); self._lock=Lock()
        self._stages={}; self._counts=Counter()
        self._frames=deque(maxlen=self.config.window)
        self._resources=deque(maxlen=min(self.config.window,120))
        self._next_resource=0; self._process=None; self._peak_rss=0
        self.last_acquisition_ns=None
        self.source_fps=None

    @contextmanager
    def span(self,name):
        if not self.enabled or not getattr(self._local,'sample',True):
            yield
            return
        if name not in STAGES: raise ValueError('Unknown profiling stage')
        stack=getattr(self._local,'stack',None)
        if stack is None: stack=[]; self._local.stack=stack
        entry=[self.clock(),0]; stack.append(entry)
        try: yield
        finally:
            elapsed=max(0,self.clock()-entry[0]); stack.pop()
            if stack: stack[-1][1]+=elapsed
            self.record(name,elapsed,max(0,elapsed-entry[1]))

    def record(self,name,elapsed,self_ns=None):
        if not self.enabled: return
        if name not in STAGES: raise ValueError('Unknown profiling stage')
        elapsed=integer(elapsed,'elapsed',0,2**63-1)
        own=elapsed if self_ns is None else integer(self_ns,'self_ns',0,elapsed)
        with self._lock:
            bucket=self._stages.setdefault(name,{'calls':0,'total_ns':0,'self_ns':0,
                'max_ns':0,'values':deque(maxlen=self.config.window),'self_values':deque(maxlen=self.config.window)})
            bucket['calls']+=1;bucket['total_ns']+=elapsed;bucket['self_ns']+=own
            bucket['max_ns']=max(bucket['max_ns'],elapsed)
            bucket['values'].append(elapsed);bucket['self_values'].append(own)

    def count(self,name,amount=1):
        if not self.enabled: return
        if name not in ('captured','encoded','jpeg_bytes','ocr_calls','published','delivered','subscriber_skipped','persistence_failures'):
            raise ValueError('Unknown performance counter')
        integer(amount,'counter increment',0,2**63-1)
        with self._lock: self._counts[name]+=amount

    def wrap(self,name,call):
        if not self.enabled: return call
        def timed(*args,**kwargs):
            with self.span(name): return call(*args,**kwargs)
        return timed

    def frames(self,iterator,index):
        if not self.enabled:
            yield from iterator
            return
        attempt=0
        try:
            while True:
                sampled=attempt%self.config.sample_every==0; self._local.sample=sampled
                tick=self.clock()
                try:
                    with self.span('pipeline'): chunk=next(iterator)
                except StopIteration: return
                finally: self._local.sample=True
                elapsed=self.clock()-tick
                if sampled:
                    with self._lock: self._frames.append({'frame_index':index(),'elapsed_ns':elapsed})
                self.sample_resources()
                attempt+=1
                yield chunk
        finally:
            if hasattr(iterator,'close'): iterator.close()

    def sample_resources(self,force=False):
        if not self.enabled: return
        now=time.perf_counter()
        if not force and now<self._next_resource: return
        self._next_resource=now+self.config.monitor_seconds
        try:
            import psutil
            if self._process is None:
                self._process=psutil.Process();self._process.cpu_percent(None)
            memory=self._process.memory_info();system=psutil.virtual_memory()
            self._peak_rss=max(self._peak_rss,memory.rss,getattr(memory,'peak_wset',0))
            sample={'rss_bytes':memory.rss,'peak_rss_bytes':self._peak_rss,
                    'cpu_percent':self._process.cpu_percent(None),'available_memory_bytes':system.available}
            # No CUDA initialization, NVIDIA subprocess, or extra dependency.
            import sys
            torch=sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                sample.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                              gpu_reserved_bytes=torch.cuda.memory_reserved(),
                              gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
                free,total=torch.cuda.mem_get_info()
                sample.update(gpu_free_bytes=free,gpu_total_bytes=total)
            with self._lock: self._resources.append(sample)
        except (ImportError,OSError,RuntimeError): pass

    def snapshot(self):
        with self._lock:
            buckets={k:dict(v,values=list(v['values']),self_values=list(v['self_values'])) for k,v in self._stages.items()}
            frames=list(self._frames);counts=dict(self._counts);resources=list(self._resources)
        stages={}
        for name,b in buckets.items():
            stages[name]=dict(summary(b['values']),calls=b['calls'],total_ms=b['total_ns']/1e6,
                mean_all_ms=b['total_ns']/b['calls']/1e6,max_all_ms=b['max_ns']/1e6,
                self_total_ms=b['self_ns']/1e6,self_latency=summary(b['self_values']))
            if name in ('subscriber_delay','acquisition_to_subscriber'):
                stages[name].update(scope='delivery_latency_overlaps_processing',self_total_ms=None,self_latency=None)
            else:
                stages[name]['scope']='processing' if name not in ('publication','lock_wait','pacing') else name
        ranking=sorted([{'stage':name,'self_total_ms':stage['self_total_ms']} for name,stage in stages.items()
                        if stage['scope']=='processing'],key=lambda row:row['self_total_ms'],reverse=True)
        return {'enabled':self.enabled,'sample_every':self.config.sample_every,'stage_accounting':'Inclusive stages nest; rank self_total_ms, do not add overlapping totals. Percentiles use the bounded recent sample window; totals/calls cover timed calls, which are sampled when sample_every > 1. Counters cover all frames/calls.',
                'stages':stages,'processing_stage_ranking':ranking,'frame_latency':summary([f['elapsed_ns'] for f in frames]),
                'recent_frames':frames,'counts':counts,'resources':resources,
                'source_fps':self.source_fps,'processing_cannot_keep_up':
                    (len(frames)*1e9/sum(max(1,f['elapsed_ns']) for f in frames)<self.source_fps) if self.source_fps and frames else None}


class FramePacer:
    """Source-order pacing only. Never samples, drops or seeks capture frames."""
    def __init__(self,mode='source',fps=None,stop=None,clock=time.perf_counter,sleep=time.sleep):
        if mode not in ('source','unpaced'): raise ValueError('Unknown pacing mode')
        self.mode=mode;self.fps=fps if isinstance(fps,(int,float)) and not isinstance(fps,bool) and math.isfinite(fps) and .1<=fps<=1000 else None
        self.stop=stop;self.clock=clock;self.sleep=sleep
        self.started=None;self.first_index=None;self.first_timestamp=None

    def pace(self,index,timestamp_ms=None):
        if self.mode=='unpaced' or self.fps is None:return False
        now=self.clock()
        valid=isinstance(timestamp_ms,(int,float)) and not isinstance(timestamp_ms,bool) and math.isfinite(timestamp_ms) and timestamp_ms>=0
        if self.started is None:
            self.started=now;self.first_index=index
            self.first_timestamp=timestamp_ms if valid else None
            return False
        elapsed=(index-self.first_index)/self.fps
        if valid and self.first_timestamp is not None and timestamp_ms>=self.first_timestamp:
            elapsed=(timestamp_ms-self.first_timestamp)/1000
        delay=self.started+elapsed-now
        if delay<=0:return False
        if self.stop is not None:return self.stop.wait(delay)
        self.sleep(delay);return False


