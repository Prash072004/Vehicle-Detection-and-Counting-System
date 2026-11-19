"""FastAPI transport and lifespan; computer vision stays in the existing pipeline."""
from contextlib import asynccontextmanager
from enum import Enum
from threading import Lock
from typing import List

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from processing_service import ProcessingService


class SignalColor(str, Enum):
    red = 'red'
    green = 'green'


def create_app(runtime):
    """Callbacks look up live pipeline state so mocks need no actual weights."""
    def service():
        return ProcessingService(lambda: runtime['_start_processing'](),
                                 lambda: runtime['generate_frames'](),
                                 lambda: runtime['shutdown_resources']())

    @asynccontextmanager
    async def lifespan(app):
        if not app.state.lifecycle_lock.acquire(blocking=False):
            raise RuntimeError('Application lifespan is already active')
        try:
            if app.state.processing.completed:
                app.state.processing = service()
            owner = app.state.processing
            try:
                await anyio.to_thread.run_sync(owner.start)
                yield
            finally:
                await anyio.to_thread.run_sync(owner.stop)
        finally:
            app.state.lifecycle_lock.release()

    app = FastAPI(title='Traffic Signal Violation Detection System', lifespan=lifespan)
    app.state.processing = service()
    app.state.lifecycle_lock = Lock()
    root = runtime['PROJECT_ROOT']
    templates = Jinja2Templates(directory=str(root / 'templates'))
    app.mount('/static', StaticFiles(directory=str(root / 'static')), name='static')

    @app.get('/', response_class=HTMLResponse)
    def index(request: Request):
        with runtime['_signal_lock']:
            light = runtime['traffic_light']
        return templates.TemplateResponse(request=request, name='index.html', context={'light': light})

    @app.get('/video', response_class=StreamingResponse,
             responses={200: {'content': {'multipart/x-mixed-replace': {}}}, 503: {'description': 'Video unavailable'}})
    def video(request: Request):
        owner = request.app.state.processing
        try:
            # Normally already ready through lifespan. Also handles explicit
            # transport-only tests without lifespan, using the same sole owner.
            owner.start()
        except Exception:
            return PlainTextResponse('Video unavailable. See server logs for details.', status_code=503)
        return StreamingResponse(owner.subscribe(), media_type='multipart/x-mixed-replace; boundary=frame')

    @app.post('/set_light/{color}', response_class=RedirectResponse, status_code=302)
    def set_light(color: SignalColor):
        with runtime['_signal_lock']:
            if runtime['traffic_light'] != color.value:
                runtime['traffic_light'] = color.value
                runtime['_signal_version'] += 1
        return RedirectResponse('/', status_code=302)

    @app.get('/violations', response_model=List[str])
    def get_violations():
        with runtime['_resource_lock']:
            return list(runtime['plate_memory'].values())

    return app
