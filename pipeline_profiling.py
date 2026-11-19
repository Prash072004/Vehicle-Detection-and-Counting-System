"""Optional owned-instance instrumentation; disabled mode installs no wrappers."""
from contextlib import contextmanager
from functools import wraps


class Proxy:
    def __init__(self,target,methods):
        self._target=target; self._methods=methods
    def __getattr__(self,name):
        return self._methods[name] if name in self._methods else getattr(self._target,name)
    def __call__(self,*args,**kwargs):
        return self._methods['__call__'](*args,**kwargs)


class TimedLock:
    def __init__(self,lock,monitor):self.lock=lock;self.monitor=monitor
    def __getattr__(self,name):return getattr(self.lock,name)
    def __enter__(self):
        with self.monitor.span('lock_wait'):self.lock.acquire()
        return self
    def __exit__(self,*args):self.lock.release()


class PipelineInstrumentation:
    def __init__(self,runtime,monitor):
        self.runtime=runtime;self.monitor=monitor;self._undo=[]

    def assign(self,target,name,value):
        if isinstance(target,dict):
            old=target[name]; target[name]=value
            self._undo.append(lambda:target.__setitem__(name,old) if target.get(name) is value else None)
        else:
            # Restore instance overrides exactly, including inherited methods.
            had=name in getattr(target,'__dict__',{});old=getattr(target,name)
            setattr(target,name,value)
            self._undo.append(lambda:setattr(target,name,old) if had else delattr(target,name))

    def method(self,target,name,stage):
        if target is not None and hasattr(target,name):
            self.assign(target,name,self.monitor.wrap(stage,getattr(target,name)))

    def install(self):
        p=self.monitor;r=self.runtime
        if not p.enabled: return self
        self.assign(r,'_resource_lock',TimedLock(r['_resource_lock'],p))
        capture=r['cap']; native_read=capture.read
        def read():
            p.last_acquisition_ns=p.clock()
            with p.span('capture'): result=native_read()
            if result[0] and result[1] is not None and result[1].size: p.count('captured')
            return result
        self.assign(r,'cap',Proxy(capture,{'read':read}))
        cv=r['cv2']; functions={}
        for name,stage in [('resize','resize'),('line','annotation'),('circle','annotation'),('rectangle','annotation'),('putText','annotation')]:
            functions[name]=p.wrap(stage,getattr(cv,name))
        native_encode=cv.imencode
        def encode(*args,**kwargs):
            with p.span('jpeg'): result=native_encode(*args,**kwargs)
            if result[0] and result[1] is not None:
                p.count('encoded');p.count('jpeg_bytes',result[1].nbytes)
            return result
        functions['imencode']=encode
        self.assign(r,'cv2',Proxy(cv,functions))
        zone=r['cvzone']
        self.assign(r,'cvzone',Proxy(zone,{n:p.wrap('annotation',getattr(zone,n)) for n in ('cornerRect','putTextRect')}))
        for name,stage in [('coco_model','vehicle_inference'),('license_plate_detector_model','plate_inference')]:
            model=r[name];self.assign(r,name,Proxy(model,{'__call__':p.wrap(stage,model)}))
        for name,stage in [('detect_vehicles','vehicle_detection'),('get_car','association'),('safe_crop','crop'),('preprocess','preprocess')]:
            self.assign(r,name,p.wrap(stage,r[name]))
        native_recognize=r['read_license_plate']
        def recognize(*args,**kwargs):
            p.count('ocr_calls')
            with p.span('ocr_pipeline'): return native_recognize(*args,**kwargs)
        self.assign(r,'read_license_plate',recognize)
        tracker=r['mot_tracker']
        self.method(tracker,'update','tracking_lifecycle')
        self.method(getattr(tracker,'sort',None),'update','sort')
        for name in ('observe_bbox','expire','mark_uncertain','enrich_event'):
            self.method(r['crossing_detector'],name,'crossing')
        ocr=r.get('ocr_manager')
        for name in ('due','expire','begin','record','invalidate'): self.method(ocr,name,'consensus')
        store=r.get('storage_service')
        if store is not None:
            native_failure=store._failure
            def failure(error):
                p.count('persistence_failures')
                return native_failure(error)
            self.assign(store,'_failure',failure)
            for name in ('before_frame','candidate','capture_plate','sync'): self.method(store,name,'storage')
            self.method(store,'_database_call','sqlite')
            for name,stage in [('save_event','event_insert'),('update_recognition','ocr_update'),('save_evidence','evidence_metadata')]:
                self.method(store.repository,name,stage)
            native_connection=store.repository.connection
            @contextmanager
            def connection(*args,**kwargs):
                with p.span('sqlite_transaction'):
                    with native_connection(*args,**kwargs) as result: yield result
            self.assign(store.repository,'connection',connection)
            self.method(store.evidence,'capture','evidence')
            # Fine-grained optional hooks consumed by the evidence writer.
            self.assign_new(store.evidence,'performance',p)
        # The reader stays lazy: only attach when existing code asks for it.
        import util
        if hasattr(util,'get_reader'):
            original_reader=util.get_reader;seen=set()
            def get_reader():
                reader=original_reader()
                if id(reader) not in seen:
                    self.method(reader,'readtext','easyocr');seen.add(id(reader))
                return reader
            self.assign(util,'get_reader',get_reader)
        native_frames=r['generate_frames']
        self.assign(r,'generate_frames',lambda:p.frames(native_frames(),lambda:r['_frame_index']))
        return self

    def assign_new(self,target,name,value):
        if hasattr(target,name): self.assign(target,name,value)
        else:
            setattr(target,name,value);self._undo.append(lambda:delattr(target,name))

    def close(self):
        for undo in reversed(self._undo): undo()
        self._undo.clear()
