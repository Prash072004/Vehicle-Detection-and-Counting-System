"""One synchronous processing owner, with a single bounded JPEG broadcast slot."""
import asyncio
import logging
import threading
import time


logger = logging.getLogger(__name__)
_process_owner = threading.Lock()


class ProcessingService:
    """Inference never runs in the ASGI loop or in a viewer's request thread.

    Startup owns the capture/models; first subscription starts frame consumption.
    After that, processing continues to EOF even with no viewers. Slow viewers
    skip old JPEGs. They never queue frames or advance the underlying source.
    """

    def __init__(self, startup, frames, shutdown, poll_interval=0.01, stop_timeout=30):
        self.startup = startup
        self.frames = frames
        self.shutdown = shutdown
        self.poll_interval = poll_interval
        self.stop_timeout = stop_timeout
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._activate = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._startup_error = None
        self._done = False
        self._latest = None
        self._sequence = 0
        self._viewers = 0
        self._waiters = {}
        self.monitor = None
        self._first_publication_ns = None
        self._last_publication_ns = None

    @property
    def completed(self):
        with self._lock:
            return self._done

    @property
    def viewers(self):
        with self._lock:
            return self._viewers

    @property
    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        with self._lock:
            if self._thread is None:
                if self._stop.is_set():
                    raise RuntimeError('Processing service has already stopped')
                if not _process_owner.acquire(blocking=False):
                    raise RuntimeError('Another application already owns video processing')
                self._thread = threading.Thread(target=self._run, name='traffic-processing', daemon=False)
                try:
                    self._thread.start()
                except BaseException:
                    self._thread = None
                    _process_owner.release()
                    raise
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError('Video processing startup failed; see server logs') from self._startup_error

    def _run(self):
        iterator = None
        try:
            self.startup()
            self._ready.set()
            self._activate.wait()
            if not self._stop.is_set():
                iterator = iter(self.frames())
                while not self._stop.is_set():
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        break
                    if self._stop.is_set():
                        break
                    monitor = self.monitor
                    tick = time.perf_counter_ns() if monitor is not None and monitor.enabled else None
                    with self._lock:
                        self._sequence += 1
                        published = time.perf_counter_ns()
                        if self._first_publication_ns is None:self._first_publication_ns=published
                        self._last_publication_ns=published
                        acquisition = monitor.last_acquisition_ns if monitor is not None and monitor.enabled else None
                        self._latest = (self._sequence, chunk, (published, acquisition))
                    self._notify()
                    if tick is not None:
                        monitor.record('publication', time.perf_counter_ns()-tick)
                        monitor.count('published')
        except BaseException as error:
            self._error = error
            if not self._ready.is_set():
                self._startup_error = error
            logger.exception('Video processing worker failed')
        finally:
            try:
                if iterator is not None and hasattr(iterator, 'close'):
                    iterator.close()
            finally:
                try:
                    self.shutdown()
                except BaseException as error:
                    self._error = error
                    logger.exception('Video processing cleanup failed')
                finally:
                    with self._lock:
                        self._done = True
                    self._notify()
                    _process_owner.release()
                    self._ready.set()

    def _notify(self):
        # At most one scheduled callback per subscriber, even if its loop is
        # stalled or its response is under network backpressure. No frame queue.
        with self._lock:
            pending=[]
            for token, waiter in self._waiters.items():
                if not waiter['queued'] and not waiter['signaled']:
                    waiter['queued']=True
                    pending.append((token,waiter['loop']))
        for token,loop in pending:
            try:
                loop.call_soon_threadsafe(self._wake, token)
            except RuntimeError:
                with self._lock:
                    self._waiters.pop(token,None)
                    self._viewers=len(self._waiters)

    def _wake(self,token):
        with self._lock:
            waiter=self._waiters.get(token)
            if waiter is None: return
            waiter['queued']=False
            waiter['signaled']=True
            event=waiter['event']
        event.set()

    async def subscribe(self):
        loop=asyncio.get_running_loop();event=asyncio.Event();token=object()
        with self._lock:
            # Deliver the latest frame immediately to a new viewer, but do not
            # count frames produced before it connected as subscriber skips.
            sequence = self._sequence if self._done or self._latest is None else self._sequence-1
            self._waiters[token]={'loop':loop,'event':event,'queued':False,'signaled':False}
            self._viewers=len(self._waiters)
        self._activate.set()
        try:
            while True:
                with self._lock:
                    event.clear()
                    waiter=self._waiters.get(token)
                    if waiter is not None: waiter['signaled']=False
                    latest, done = self._latest, self._done
                if latest is not None and latest[0] > sequence:
                    monitor=self.monitor
                    if monitor is not None and monitor.enabled:
                        now=time.perf_counter_ns()
                        monitor.record('subscriber_delay',max(0,now-latest[2][0]))
                        if latest[2][1] is not None:
                            monitor.record('acquisition_to_subscriber',max(0,now-latest[2][1]))
                        monitor.count('delivered')
                        monitor.count('subscriber_skipped',max(0,latest[0]-sequence-1))
                    sequence = latest[0]
                    yield latest[1]
                elif done:
                    return
                else:
                    await event.wait()
        finally:
            with self._lock:
                self._waiters.pop(token,None)
                self._viewers=len(self._waiters)

    def stop(self):
        self._stop.set()
        self._activate.set()
        thread = self._thread
        if thread is not None:
            if thread is threading.current_thread():
                raise RuntimeError('Processing owner cannot join itself')
            thread.join(self.stop_timeout)
            if thread.is_alive():
                # Never release a capture underneath native inference, or allow
                # a second owner to start while the old owner is still running.
                raise RuntimeError('Processing worker did not stop within the shutdown timeout')

    def snapshot(self):
        """Private operational state; deliberately not an HTTP endpoint."""
        with self._lock:
            state={'active_viewers':self._viewers,'publication_count':self._sequence,
                   'buffered_frames':int(self._latest is not None),'processing_queue_depth':0,
                   'notification_callbacks_pending':sum(int(w['queued']) for w in self._waiters.values()),
                   'detection_frames_skipped':0,'completed':self._done}
            duration=(self._last_publication_ns-self._first_publication_ns) if self._sequence>1 else 0
            state['processing_fps']=((self._sequence-1)*1e9/duration) if duration>0 else None
            state['source_fps']=self.monitor.source_fps if self.monitor is not None else None
            state['processing_cannot_keep_up']=(state['processing_fps']<state['source_fps']) if state['source_fps'] and state['processing_fps'] is not None else None
        if self.monitor is not None:state['performance']=self.monitor.snapshot()
        return state
