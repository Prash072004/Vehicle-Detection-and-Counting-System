# Phase 9: traffic-monitoring dashboard

Baseline: `e411ef9` on `main`, with a clean working tree and 502 passing tests.
HTML, CSS and vanilla JavaScript are retained. No frontend framework, package
installation, branch, commit or push is part of this phase.

## Interface

The charcoal dashboard gives the 3:2 road view priority, with a simulated
red/green signal panel and session details beside it. Six compact metric cards
explain their units and counting semantics. Below are recent candidate summaries,
current-session accepted OCR strings, and a restricted evidence preview panel.
There are no decorative analytics, fake counts, yellow controls or automatic
traffic-light recognition.

Desktop uses a road/control grid. Below 850px, monitoring stacks above a
two-column control area; below 540px, panels use one column and metrics use two.
The road image uses `object-fit: contain`, a 3:2 stage and its native 720x480
processing ratio, preserving all annotations. Tables scroll within their panel.
System fonts and local assets require no external font or CDN requests.

Semantic landmarks, one h1, section headings, table headings, form buttons,
labels, image descriptions, a skip link, visible keyboard focus and restrained
polite status regions support keyboard and assistive-technology use. Signal
state has text as well as color; buttons expose `aria-pressed`. Reduced-motion
styles suppress animation. The native evidence dialog supports Escape, focus
containment and returning focus to its opener in supporting browsers. Metrics
are not individually live-announced every poll.

## Endpoints and contracts

| Endpoint | Contract |
| --- | --- |
| `GET /` | New dashboard, rendered with current signal state |
| `GET /static/...` | Local CSS, JS and explicitly synthetic SVG fixture only |
| `GET /video` | Existing shared MJPEG; boundary and startup 503 behavior preserved |
| `POST /set_light/red`, `/green` | Existing 302 redirect to `/`; invalid colors remain 422 |
| `GET /violations` | Existing current-process list of accepted plate strings; now `Cache-Control: no-store` |
| `GET /api/status` | New typed, read-only monitoring snapshot, `Cache-Control: no-store` |
| `/docs`, `/openapi.json` | Existing documentation, including the new status schema |

`DashboardStatus` and its nested Pydantic models document the complete status
response in OpenAPI. Example before startup:

```json
{
  "schema_version": 1,
  "state": "not_started",
  "session": null,
  "signal": {"color": "green", "version": 0},
  "source": {"kind": "unavailable", "width": null, "height": null, "total_frames": null},
  "metrics": {
    "processed_frames": null, "frame_index": null, "processing_fps": null,
    "source_fps": null, "vehicle_detections": null, "observed_tracks": null,
    "candidates": null, "accepted_plates": null, "active_viewers": 0
  },
  "recent_candidates": [],
  "storage": {"state": "unavailable", "evidence_capture": false},
  "access": {"history": false, "evidence": false},
  "error": null
}
```

The endpoint never initializes models, consumes frames or queries SQLite. The
owner thread publishes one bounded monitoring snapshot per successfully encoded
frame. A separate short lock protects scalar counts and at most 20 small rows.
Requests copy this snapshot and read the producer's small transport state and
signal lock; they do not acquire the inference/resource lock. Profiling reports
and resource details remain private. No new processing thread is introduced.

## Metrics and state

| Field | Source and meaning |
| --- | --- |
| `processed_frames` | Successfully processed and JPEG-encoded frames, once per source frame, independent of viewers |
| `frame_index` | Original zero-based source index of the last successfully encoded frame |
| `processing_fps` | Completed frames divided by their accumulated read/processing/encoding seconds; excludes pacing, viewer/network time, startup and EOF read |
| `source_fps` | Existing validated capture FPS; unknown values remain null |
| `vehicle_detections` | Sum of validated vehicle rows on completed frames; repeated observations count again |
| `observed_tracks` | Confirmed, measurement-updated tracking rows in the last completed frame; not unique physical vehicles or predicted/lost tracks |
| `candidates` | Current crossing detector's event count, including any subsequently uncertain candidates |
| `accepted_plates` | Size of current `plate_memory`, not distinct registrations or human verification |
| `active_viewers` | Registered subscribers to the existing shared producer |

Failures before a completed frame do not invent statistics. Initialized counters
start at zero, FPS stays null until measured, and missing metrics display an em
dash or “Unavailable”. Last-frame counts remain visible on completion; they do
not imply vehicles are still active. JPEG failures are not counted as completed
frames. Source progress uses `(frame_index+1)/total_frames`, labeled as source
frames reached; unknown lengths have no percentage. This is not seeking support.

`session` is a fresh opaque **monitoring token**, not the SQLite session UUID.
It resets on a new processing lifespan. Source labels expose only local-video
or camera kind, dimensions and optional frame count, never a path or filename.

States combine actual producer activation/readiness/stopping/error flags with
explicit pipeline EOF/failure outcomes: not_started, initializing, ready,
processing, completed, stopping, stopped, error. Errors are fixed safe messages,
never Python exceptions or paths. Startup failure of the whole ASGI lifespan
can prevent the server from serving the page at all; the browser then reports
connection failure. It cannot distinguish model loading from source opening
inside that startup, so the UI does not invent finer progress stages.

Storage is disabled, unavailable, ready or degraded based on the existing
storage object's enabled flag, initialized-session flag and recorded failures.
“Degraded” is conservative for the session after a failure; it is not a database
health probe or proof every candidate was saved. Evidence-capture configuration
does not imply evidence exists for any candidate.

## Signal and networking behavior

Both native POST forms remain usable without JS. With JS, only one control
request is allowed at a time. Both buttons disable while pending; the current
signal is not optimistically changed. After the POST and its compatibility
redirect, a status request confirms the server's actual signal/version, including
changes by another viewer. Failure retains the last confirmed text, reports the
uncertainty and disables controls until monitoring reconnects. Repeated active
color submissions are disabled. Request revisions and signal versions prevent
stale polling responses from overwriting newer confirmed state.

Status polling schedules the next request **2 seconds after the current update
finishes**, normally no more than 0.5 requests/second per visible viewer. Polls
never overlap. Accepted strings are fetched only when the real accepted count
is positive; unchanged results refresh after 10 seconds, count changes refresh
earlier. Session changes clear old plate DOM. Up to 100 strings are rendered;
the existing list endpoint itself remains unchanged. Control confirmation adds
one status request and the existing redirect adds a homepage GET.

Requests use HTTP-status checks, schema/value checks, `no-store` and an
8-second AbortController timeout. Failed updates show stale-data feedback.
Hidden pages cancel pending monitoring requests and timers; showing the page
resumes polling. Page lifecycle cleanup removes listeners and cancels work,
including safe handling of back/forward caching. Hiding/disconnecting a viewer
does not stop the shared producer. No WebSocket, automatic reconnect loop,
additional MJPEG element, model instance or JPEG encoder was added.

The image subscribes once after backend readiness. Processing state comes from
the server, never solely from image events. A stream error offers an explicit
reconnect to the same producer; it does not restart processing. Completion and
terminal errors have their own messages. A newly opened page after EOF cannot
recover the final JPEG from the completed subscription; it shows completion
without pretending to have live footage. No pause, stop, restart, source switch
or seeking UI is offered because the existing backend has no safe per-viewer
contract for these operations.

## Candidate, plate and evidence boundaries

The latest 20 **created current-session in-memory** candidates expose only track
ID, crossing frame, signal, candidate/uncertain state and OCR status. No plate
text, event UUID, crop, source path, persistence diagnostics or review notes are
copied into `/api/status`. Accepted, unreadable, pending, not attempted,
conflicting, low confidence and unrecognized-format states remain distinct.
Rows are ordered by creation, not sorted on every poll. A candidate remains a
candidate after accepted OCR. Human review state is deliberately not loaded.

Historical records, real evidence and review controls remain inaccessible.
There are no new database-history, file-serving or review endpoints. `/violations`
retains its pre-existing current-session privacy boundary; it is unauthenticated
and is appropriate only within the existing trusted local deployment. This phase
does not add production authentication or justify public internet exposure.

The evidence dialog is a **synthetic component preview**, available only through
an explicitly labeled button. Its fixed local SVG is geometric test data, not
road imagery or evidence. It demonstrates type/frame labels, contained aspect
ratio, a missing-plate option and an image-error state. The browser never supplies
a server filesystem path. No actual saved evidence is opened, changed or served.

## Verification

```powershell
.\myvenv\Scripts\python.exe -B -m unittest discover -s tests -v
node --test tests/frontend.test.cjs
```

All 502 earlier tests are retained; the exact OpenAPI path-set assertion includes
the new endpoint. Twenty-eight new Python checks cover status schema/privacy,
signal compatibility, a real mocked pipeline's counts and one capture release,
bounded/atomic snapshots, readiness/EOF/failure, unavailable metrics, static
fixture restrictions, semantic/responsive markup and Node integration.
Twenty-two Node tests use built-in `node:test` and a lightweight DOM fixture to
exercise request errors/timeouts/cancellation, non-overlap, stale signals,
server confirmation, empty states, plate throttling, stream reconnect policy,
page lifecycle and evidence preview. If Node is absent only its Python wrapper
is skipped; no install is attempted. These tests do not claim visual layout or
assistive-technology certification.

Final result: **530 Python tests passed, zero failures, zero skips**, including
all 502 previous checks and the 28 new checks. The included Node wrapper passed
all **22 JavaScript tests**.
Syntax parsing passed for all 42 Python source/test/tool files; JavaScript
syntax checks passed. Logs remain in ignored `outputs/phase9_final_tests.log`
and `outputs/phase9_js_tests.log`.

### Actual browser verification

Chrome 157 was driven locally through its debugging protocol with existing Node
and Python; no automation package was installed. The real FastAPI application,
existing sample video, YOLO models and SORT produced the displayed annotated
frames. Database and evidence writes were disabled **only in the test process**.
Cold native SciPy/BLAS loading on the producer thread stalled during early smoke
attempts. The verification harness preloaded the existing SORT/SciPy module on
its main thread before launching Uvicorn. Production startup and library settings
were not changed to work around this host-specific native-import limitation.

Verified at 1440x1080 desktop, 768x1024 tablet and 390x844 mobile viewports:

- No document-level horizontal overflow; the full 720x480 stream maintained 3:2
  dimensions (360x240 on mobile), without cropping or stretching road regions.
- Actual red-signal POST/302/confirmation and the updated red text/indicator.
- Two subscribers, exactly two browser video requests (one per page), a constant
  monitoring session token, and continued processing after one viewer closed.
- Real status-request blocking/recovery in the browser, stale-data feedback and
  resumed successful updates, without an automatic video reconnect.
- Synthetic dialog image, native Escape, restored opener focus and a visible
  keyboard-focus outline. Foreground tabs were used for focus/network checks.
- Zero uncaught JavaScript exceptions. Status polls were spaced about 2.0-3.5
  seconds during the measured observation; signal confirmation added one nearby
  request. The hidden tab stopped regular monitoring polls.

Screenshots, all ignored and local:

- `outputs/phase9_desktop.png`
- `outputs/phase9_tablet.png`
- `outputs/phase9_mobile.png`
- `outputs/phase9_evidence_preview.png` (explicitly synthetic preview)

`outputs/phase9_browser_results.json` records assertions and request timestamps.
The final observed session reached 35 completed frames, 781 validated vehicle
observations and 20 confirmed observed tracks in the last frame. No crossing
candidate or accepted OCR result existed at that observation point; the empty
states reflect those actual values. This short browser run does not establish
OCR accuracy or exercise the recording's entire duration. Completion, candidates,
OCR acceptance, inference failures and persistence behavior have deterministic
automated coverage; they are not all claimed as visually observed scenarios.
Screen-reader use, physical mobile devices and long-running camera behavior
still need manual validation.

### Cost and regression audit

Twenty loopback `/api/status` requests during live inference measured 11.38 ms
median and 26.87 ms maximum wall latency. An isolated five-repeat benchmark of
1,000 updates/reads, with 500 available events and only 20 copied summaries,
measured about 31.54 microseconds per frame observer update and 259.21 microseconds
per snapshot copy (medians). These host-dependent measurements are not an
inference-FPS comparison or a general latency guarantee. No frontend FPS speedup
is claimed. The response path performs no database scan or inference-lock wait.
See ignored `outputs/phase9_observer_cost.json` for the measurement scope.

Only eight intended existing files changed: README, main, processing service,
web API, the HTML/CSS/JS frontend and one OpenAPI assertion in `test_fastapi.py`.
Five files were added: dashboard state/schema, synthetic SVG fixture, Python
tests, Node tests and this document. No baseline file was deleted. SHA-256 hashes
of all other original files, including models, videos and algorithm modules,
are unchanged. All **46,834 virtual-environment files** retained their count,
size and modification-time digest. Generated browser profiles, temporary library
settings, screenshots and logs remain under ignored `outputs/`; no production
database was created. No packages were installed or changed.

The branch remains `main` at `e411ef9`. All 13 Phase 9 files remain unstaged and
uncommitted, with an empty index. No branch, commit, push or remote change occurred.

## Manual browser checklist

1. Start the single-worker server as described in README. Inspect the page at
   1440px, 768px, 390px and 320px. Confirm no document-level horizontal overflow
   and the full 3:2 road image remains visible.
2. Watch first-frame readiness, progress, actual counters and video completion.
   Unknown-FPS/camera sources should not acquire fabricated lengths or percentages.
3. Switch red/green with keyboard and pointer. Throttle or block a request in
   developer tools: the UI must retain the last confirmed signal and explain
   failure. Invalid POST colors must still return 422.
4. Open two viewers. Verify one producer and increasing source counters, then
   close one viewer while the other continues. Check the Network panel for one
   video request per page and normally no more than one status poll per 2 seconds.
5. Hide/show a tab, simulate offline status, recover connectivity and interrupt
   the image stream. Confirm no automatic stream reconnect loop.
6. Tab through controls, check visible focus and the skip link. Open the synthetic
   evidence preview; test Escape, returned focus, the missing option and a blocked
   fixture request. No real evidence should load.
7. Check no-candidate/no-accepted-plate states and inspect the distinction between
   candidate, unreadable, accepted OCR and human review being unavailable.
8. Test with a screen reader and at 200% zoom before deployment; automated markup
   and headless browser checks alone cannot certify accessibility.
