# Phase 7: Flask to FastAPI migration

This phase replaces the HTTP framework and streaming ownership. It preserves
the six previous detection, tracking, crossing, OCR and persistence phases.
Work remains unstaged on `main`; no dependency installation, commit or push is
part of this migration.

## Starting state and unexpected artifact

Baseline: `8626582c07105a815a892c043e69ee06ea1d441f`,
`Implement persistent violation records and evidence management`.
The original suite passed all 412 tests before editing.

An untracked 902-byte file named `istent violation records and evidence
management` followed by U+F022 contained only the latest commit's colored
`git show --stat` output. Removing ANSI color sequences and normalizing line
endings reproduced that output exactly. No source referenced it. It was removed
using its exact literal path, after checking it was a regular file, leaving a
clean working tree before migration. It was probably terminal redirection from
a malformed command; the originating command cannot be proved from the file.
No unique project data was present. No other existing file was deleted.

## File responsibilities

- `main.py`: application entry point, resource lifecycle and existing pipeline
  functions. Their public names remain available to earlier integrations/tests.
- `web_api.py`: FastAPI factory, lifespan, typed HTTP endpoints, Jinja templates
  and static mounting. It receives live runtime state/callbacks, so tests can
  replace models without loading real weights.
- `processing_service.py`: sole worker owner and bounded latest-JPEG broadcast.
- `templates/index.html`: only the two static `url_for` arguments change from
  Flask's `filename` to Starlette's `path`. HTML layout, CSS and JS are retained.
- `requirements.txt`: exact installed dependency closure for the tested runtime.
- `tests/test_fastapi.py`: 33 additional framework and producer checks.
- Existing tests: framework-specific client, route-inspection and server-start
  assertions use FastAPI/HTTPX/Uvicorn. CV assertions remain in place.

The existing CV functions deliberately remain accessible in `main.py` instead
of forcing changes to existing callers and regression fixtures. No inference
runs in a route handler. Python AST comparison against the baseline verifies
all ten pre-existing resource/pipeline functions are unchanged, including
`generate_frames`, initialization/shutdown, timestamp handling and storage/OCR
accessors. Vehicle, tracking, crossing, plate, SQLite, evidence, export and SORT
source files are unchanged. SORT attribution and licensing remain intact.

## Routes and frontend compatibility

| Route | Behavior |
| --- | --- |
| `GET /` | Existing Jinja HTML with current signal, absolute asset directory |
| `GET /static/...` | Existing CSS/JS through `StaticFiles` |
| `GET /video` | `StreamingResponse`, `multipart/x-mixed-replace; boundary=frame` |
| `POST /set_light/red` | Set red and increment version only on transition; 302 `/` |
| `POST /set_light/green` | Set green and increment version only on transition; 302 `/` |
| `POST /set_light/{unsupported}` | Enum/Pydantic validation, 422, no state change |
| `GET /violations` | Typed JSON list of accepted plate strings from this session |
| `GET /docs`, `/redoc`, `/openapi.json` | Automatic API documentation/schema |

Each MJPEG part remains `--frame\r\nContent-Type: image/jpeg\r\n\r\n`, JPEG bytes,
then `\r\n`. The video element and 500-ms JS polling are unchanged. No history,
review, raw candidate, evidence file or database endpoint is exposed. Static
mounting includes only `static/`, not the project root. Unsupported HTTP methods
retain framework 405 handling. Invalid colors now return an explicit validation
error instead of Flask's previous silent redirect.

## Lifespan and ownership

FastAPI's async context-manager lifespan calls `ProcessingService.start` through
AnyIO's thread runner; initialization and processing themselves execute on the
same dedicated, non-daemon `traffic-processing` thread. The worker owns model
loading, capture reads, tracker updates, crossing/OCR, storage and final cleanup.
It initializes once, signals readiness, then waits until the first viewer
subscribes before consuming video. Two YOLO models load once per lifespan;
EasyOCR retains its original lazy, cached reader. Imports start no worker/model.

Lifespan startup failures fail server startup and release partial resources.
The route's defensive initialization path returns 503 when used without a
successful lifespan (also exercised by transport-only tests). Normal deployment
must retain Uvicorn lifespan handling; do not launch with `--lifespan off`.

A process-level ownership lock rejects a second processing service. A per-app
lifespan lock rejects overlapping startup/shutdown. A subsequent completed
lifespan can create a fresh service and session, clear previous process-local
results, and reopen resources. This does not delete historical SQLite records.

There is one latest encoded-frame slot, not a per-client queue. Viewers use
async iteration and brief 10-ms polling to read newer sequence numbers. They
never call capture/tracker/persistence themselves. Slow viewers skip old frames;
network backpressure can retain an in-flight frame per connection, not an
unbounded history. One disconnect cancels only that viewer's async iterator.
After first activation, processing continues to EOF even with zero viewers.

EOF preserves Phase 6 `completed`; processing errors preserve `failed`; server
shutdown before EOF uses `stopped`. Existing JPEG failure limits remain intact.
The last available JPEG can be delivered before an active stream completes;
new subscriptions after completion are empty, with no replay or implicit loop.

Shutdown requests stop, wakes an idle worker, and joins it off the ASGI event
loop. The generator/capture and storage finish on their owner thread. Native
OpenCV/Torch/OCR work cannot safely be forcibly interrupted. If it fails to
return within 30 seconds, shutdown raises an explicit timeout and retains
ownership rather than releasing a capture under running inference. Such a
stuck native call needs operator intervention; a clean shutdown is not claimed.

The direct launcher also sets Uvicorn's graceful HTTP shutdown timeout to five
seconds. The documented CLI includes the same setting: Uvicorn otherwise waits
for open streaming requests before lifespan shutdown, which could wait forever
on a webcam. After this deadline it cancels open HTTP tasks so lifespan can stop
the producer; the separate 30-second native-worker join limit still applies.

HTTP route handlers that acquire existing locks run in FastAPI's thread pool,
not the event loop. Signal changes and templates use the brief signal lock.
The plate list uses the pipeline resource lock for a consistent snapshot; it
can wait for an in-progress inference frame. No SQLite connection crosses
threads: the unchanged database layer opens/closes a connection per operation.

## Persistence and evidence

Schema version 1 and the five existing tables are unchanged. Candidate insertion
still occurs before OCR; accepted OCR enrichment, review audit, stable event
UUIDs, export filters, evidence hashes and retention controls remain unchanged.
Multiple viewers do not create extra sessions or writes. Evidence is generated
once by the processing owner, independently of subscriber count.

Storage defaults to `data/violations.db`, evidence defaults off. All existing
`TRAFFIC_*` configuration remains valid; see [Phase 3](phase3_detection.md),
[Phase 4](phase4_tracking.md), [Phase 5](phase5_ocr.md) and
[Phase 6](phase6_storage.md). Configuration is read at startup. Tests use temporary
databases or ignored `outputs/` directories and never production records.

## Runtime and dependency findings

Only Python 3.8.10 was usable on this machine. `py -0p` found no registered
interpreters; discovered `python3.13.exe` was a broken Chocolatey shim targeting
missing `C:\Python313\python.exe`. No Python 3.11/3.12 environment was available.
The existing environment already contained a compatible stack:

| Package | Tested version |
| --- | --- |
| Python | 3.8.10 |
| FastAPI / Starlette | 0.120.4 / 0.44.0 |
| Pydantic / Uvicorn | 2.10.6 / 0.33.0 |
| HTTPX / AnyIO / Jinja2 | 0.28.1 / 4.5.2 / 3.1.6 |
| PyTorch / torchvision | 2.4.1+cu121 / 0.19.1+cu121 |
| Ultralytics / EasyOCR | 8.3.217 / 1.7.2 |
| OpenCV / NumPy | 4.12.0.88 / 1.24.1 |

Installed `Requires-Python` and dependency metadata support this combination;
`pip check` reports no broken requirements. FastAPI import and `TestClient`
worked before editing. No packages, interpreter or ML dependencies were added,
removed or upgraded, and `myvenv` was not modified.

The 66-distribution `requirements.txt` captures the installed Windows/Python 3.8
dependency closure, excluding unused Flask/Streamlit. It includes optional
offline-tool imports such as pandas and plotting packages. Existing CUDA-tagged
Torch wheels must be provisioned first from the PyTorch cu121 index; CPU
execution is supported with these installed wheels. Reproduction commands are
in [README](../README.md). No fresh installation was attempted, so the pinned
metadata/import tests are not a claim that a clean-room rebuild was performed.

Python 3.11/3.12 is preferred for a separately approved future environment.
Python 3.8 is an old compatibility target. FastAPI dropped 3.8 in 0.125.0;
Uvicorn dropped it in 0.34.0. The exact tested pins must not be upgraded blindly.
OpenCV 4.12's metadata requires NumPy >=2 on Python >=3.9; this file's NumPy
1.24.1 pin is therefore deliberately scoped to Python 3.8. A modern-runtime
port must choose and validate compatible OpenCV/NumPy/Torch/OCR versions with
permission before installation. No modern-runtime validation is claimed.

## Launch and tests

```powershell
.\myvenv\Scripts\python.exe -B -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1 --timeout-graceful-shutdown 5
# Equivalent local entry point:
.\myvenv\Scripts\python.exe -B main.py
# Automated regressions:
.\myvenv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Open `http://127.0.0.1:8000/`. Use one process, one worker and no reload. The
process lock cannot coordinate multiple OS processes: multiple Uvicorn workers
would each open an independent source/session. They require a separately
designed shared processing service; this migration adds no distributed system.

Automated coverage: the 412 previous tests plus 33 new checks, 445 total.
New tests cover lifespan/model initialization and failure, resource ownership,
restart/overlap, templates/static paths and bytes, typed signal/JSON/OpenAPI,
MJPEG framing/EOF, bounded slow-client behavior, cancellation, two simultaneous
ASGI viewers and an independent signal request while the producer is blocked.
A second two-subscriber test runs the actual mocked CV pipeline, confirms one
session/event/OCR invocation, and validates saved evidence hashes. Persistence
before OCR and completed/stopped session statuses are explicitly checked.
No test requires a real YOLO inference or OCR model download.

## Real-model verification

Final results are recorded in ignored `outputs/phase7_real_model_smoke_final.json`.
The earlier `phase7_real_model_smoke.json` is a preliminary run. The comparison executes the committed
Flask source with its committed template, and the working FastAPI source with
its current template, against the same first 60 source frames. A test-only
capture wrapper reports EOF at frame 60. It does not edit the recording.

Both runs use cached weights, CPU, three YOLO warmups per model, red signal,
default configurations and separate test databases. Network connections are
denied. Frame processing excludes model startup/warmup/shutdown. Delivery timing
measures JPEG encoding completion to in-process response consumption/send;
it does not measure network, proxy, browser decode or display latency. Runs are
sequential and are not evidence of a framework speedup. Concurrent-viewer tests
are separate deterministic tests, not mixed into this FPS comparison.

The final high-resolution `time.perf_counter` run measured:

| Measurement | Committed Flask | FastAPI |
| --- | --- | --- |
| Frames processed / delivered | 60 / 60 | 60 / 60 |
| Processing seconds / FPS | 18.988 / 3.16 | 30.867 / 1.94 |
| Mean / median frame latency, ms | 316.10 / 257.97 | 514.17 / 262.30 |
| p95 frame latency, ms | 464.50 | 1229.12 |
| Mean encode-to-response delivery, ms | 0.36 | 8.14 |
| p95 encode-to-response delivery, ms | 0.51 | 15.25 |
| Vehicle detections / plate inference calls | 1304 / 21 | 1304 / 21 |
| OCR calls / accepted plates | 6 / 0 | 6 / 0 |
| Persisted candidates / SQLite sessions | 2 / 1 | 2 / 1 |
| Final session / capture releases | completed / 1 | completed / 1 |

Both runs had identical validated vehicle arrays and crossing frame/point/time/
signal signatures. Candidate/event counts matched persisted rows. No plates
were accepted; no OCR accuracy claim follows. One thread owned processing in each run; FastAPI's worker
stopped after lifespan shutdown. Existing OCR quality remains a limitation.

Timing varied: an earlier complete comparison measured 3.19 FPS Flask and 2.91
FPS FastAPI. That preliminary run used Windows' coarse monotonic clock, so only
the final performance-counter run is used for detailed latency. The final
FastAPI run had larger latency spikes despite similar median frame times.
Observed throughput was lower, not faster; these sequential local samples do
not isolate the cause from host load/thread scheduling. Higher or predictable
throughput requires a separate controlled performance investigation. Neither
run measures production capacity or browser/network latency.

Hardware: Windows build 26200, AMD64 Family 23 Model 24, eight logical CPUs,
PyTorch seven CPU threads, CPU inference, the versions listed above. Startup,
warmup, independent route checks and shutdown were excluded from frame timing.
The report helper refused to overwrite the preliminary JSON only after both
final runs had completed all assertions; final per-run JSON records were
recovered verbatim from the run log into a separate new final report. No
validation was bypassed and no rerun result was fabricated.

A separate live Uvicorn/HTTPX loopback smoke test used mocked inference and real
JPEGs. Two simultaneous HTTP viewers shared one initialization and two produced
frames; viewer A disconnected after frame one while viewer B received both.
Homepage, CSS, docs, 302 signal changes, 422 invalid colors and JSON plate list
worked through the running server. Both server and producer threads stopped.
See ignored `outputs/phase7_uvicorn_http_smoke.json`. This transport test is
separate from the real-model ASGI timing and is not a real-model load test.

An additional live Uvicorn test left an open-ended stream connected during
shutdown. With a shortened test graceful timeout of 0.2 seconds, the HTTP task
was cancelled and both server and producer stopped in 0.34 seconds, with zero
remaining viewers. The application launcher uses five seconds, not the test's
shortened deadline. See ignored `outputs/phase7_shutdown_smoke.json`.

Final automated result: **445 tests passed, zero failures** (412 preserved plus
33 new). Syntax checks, dependency constraints and `git diff --check` passed.
The application imports with Flask explicitly blocked. No production database
was created. The protected-file audit found changes only to the nine intended
existing source/template/documentation/test files, and no other baseline file
changed or disappeared. All 46,834 virtual-environment files retained their
original count, sizes and modification times. Model/video/dataset assets and
SORT licensing files retained their SHA-256 hashes. The migration remains
unstaged and uncommitted on `main` at the original Phase 6 commit.

## Remaining limitations

Detection accuracy, OCR readability and SORT identity limitations are unchanged.
Shared latest-frame broadcast improves ownership, not model quality. The real
sample may produce candidates without readable plates; unit OCR fixtures are
not field accuracy evidence. Candidate memory and SQLite durability/outbox
limitations from Phase 6 remain. There is no new authentication, real-signal
synchronization, legal enforcement, production hosting or multi-worker service.
No live browser UI, WAN load, long-running camera or modern-runtime rebuild is
claimed by these tests.

## Primary references

- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)
- [FastAPI templates and static URLs](https://fastapi.tiangolo.com/advanced/templates/)
- [FastAPI release notes, Python compatibility](https://fastapi.tiangolo.com/release-notes/)
- [Uvicorn release notes, Python compatibility](https://uvicorn.dev/release-notes/)
