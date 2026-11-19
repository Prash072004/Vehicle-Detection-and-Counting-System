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
                for chunk in iterator:
                    if self._stop.is_set():
                        break
                    with self._lock:
                        self._sequence += 1
                        self._latest = (self._sequence, chunk, time.monotonic())
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
                    _process_owner.release()
                    self._ready.set()

    async def subscribe(self):
        # No blocking condition wait occupies an AnyIO thread per viewer. This
        # brief polling targets 10 ms between checks; actual delivery also
        # depends on event-loop scheduling and client/network backpressure.
        with self._lock:
            sequence = self._sequence if self._done else 0
            self._viewers += 1
        self._activate.set()
        try:
            while True:
                with self._lock:
                    latest, done = self._latest, self._done
                if latest is not None and latest[0] > sequence:
                    sequence = latest[0]
                    yield latest[1]
                elif done:
                    return
                else:
                    await asyncio.sleep(self.poll_interval)
        finally:
            with self._lock:
                self._viewers -= 1

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
