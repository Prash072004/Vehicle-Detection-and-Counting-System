# Traffic Signal Violation Detection System

A FastAPI-based computer vision prototype that detects vehicles in a prerecorded
traffic video, tracks them with SORT, and creates a candidate when a vehicle
crosses a finite stop line in the prohibited direction during a simulated red
signal. License plate detection and EasyOCR can enrich that candidate.

The signal is controlled manually from the web interface. It is not synchronized
with a real traffic-light controller or the original timing of the recorded scene.
Candidates are observations for review, not legally confirmed violations.

## Implemented features

- YOLOv8 vehicle detection for cars, motorcycles, buses, and trucks.
- Configurable confidence/inference size, validated/clipped vehicle boxes,
  optional rectangular ROI, and clean images for vehicle/plate inference.
- Startup selection of local video files or a webcam; standalone local-model
  speed benchmarking and labeled-frame accuracy evaluation.
- SORT tracking with the existing Kalman-filter and IoU association implementation.
- Configurable SORT parameters, bounded track lifecycle metadata, and confirmed
  observation-only crossing inputs. Optional headless tracking replay/evaluation.
- Stateful movement-segment intersection with a configurable finite stop line.
- Direction checks, jitter hysteresis, observation-gap and position-jump limits,
  and conservative handling of unknown or changing signal timing.
- One recorded crossing per SORT ID per processing session, with transient
  position histories kept separately from candidate events.
- Candidate IDs, crossing frame, intersection point, signal snapshot, and an
  interpolated video timestamp when usable advancing metadata is available.
- Plate detection and English OCR with configurable Indian/legacy syntax,
  confidence ranking, safe crops, conservative association, bounded event
  consensus and retry scheduling. Candidates persist when OCR returns no result.
- SQLite candidate records with processing sessions, immutable crossing metadata,
  accepted OCR enrichment, local review/correction history, filtering and exports.
- Optional bounded clean-frame evidence capture with relative references and
  integrity hashes; disabled by default. Retention is an explicit local operation.
- A browser MJPEG stream, red/green controls, and a list of recognized plates.
- Asset-path validation, bounded JPEG failure handling, EOF checks, diagnostic
  logging, and process-owned video capture cleanup. A single processing thread
  broadcasts the latest JPEG to all viewers; Uvicorn uses one worker without reload.

A green crossing is not reclassified when the light later turns red. A first
sighting already beyond the line is insufficient to establish a crossing.
Crossings with uncertain signal timing do not create red-light candidates.

## Source layout

```text
.
|-- main.py                         Entry point and compatible CV pipeline API
|-- web_api.py                      FastAPI routes, templates and lifespan
|-- processing_service.py           Single processing owner, bounded broadcast
|-- requirements.txt                Tested Windows/Python 3.8 dependency pins
|-- vehicle_detector.py             Vehicle configuration, filtering, ROI
|-- tracking_manager.py             SORT settings, lifecycle, observed outputs
|-- plate_recognition.py            Plate syntax, crops, association, event OCR
|-- database.py                     SQLite schema, records, review and queries
|-- violation_storage.py            Session persistence, retries and retention
|-- evidence_manager.py             Bounded clean-frame evidence capture
|-- violation_logic.py              Geometry, configuration, track/event state
|-- util.py                         Plate association, OCR and utility helpers
|-- add_missing_data.py             Standalone CSV interpolation helper
|-- visualize.py                    Standalone annotated-video helper
|-- sort/
|   |-- sort.py                     Vendored tracker implementation
|   |-- requirements.txt            Historical tracker-only dependencies
|   |-- README.md
|   `-- LICENSE
|-- templates/index.html
|-- static/
|   |-- style.css
|   `-- js/script.js
|-- tools/
|   |-- benchmark_detection.py       Optional local model speed comparisons
|   |-- evaluate_detection.py        Optional labeled-frame precision/recall
|   |-- benchmark_tracking.py        Fixed-detection parameter replay
|   |-- evaluate_tracking.py         Labeled identity metrics
|   |-- benchmark_ocr.py             Identical-crop preprocessing comparisons
|   |-- evaluate_ocr.py              Labeled crop exact match, CER, acceptance
|   |-- export_violations.py         Selected local records, JSON/CSV
|   `-- detection_common.py          Offline tool helpers
|-- docs/phase3_detection.md         Configuration, benchmarking, dataset format
|-- docs/phase4_tracking.md          Tracking lifecycle, metrics and experiments
|-- docs/phase5_ocr.md               OCR settings, evidence, diagnostics and tools
|-- docs/phase6_storage.md           SQLite, evidence, review, export and retention
|-- docs/phase7_fastapi_migration.md  ASGI lifecycle, viewers, setup and verification
`-- tests/
    |-- test_stability.py            34 stability checks
    |-- test_violation_logic.py      52 geometry/integration checks
    |-- test_vehicle_detection.py    Detection and application regressions
    |-- test_detection_tools.py      Offline detection tools
    |-- test_tracking.py            SORT safety, lifecycle and integration
    |-- test_tracking_tools.py      Identity metrics and fixed-input replays
    |-- test_plate_recognition.py   OCR rules, geometry, evidence, integration
    |-- test_ocr_tools.py           Known-label metrics and offline crop replay
    |-- test_database.py            Schema, transactions, records and review
    |-- test_violation_storage.py   Session persistence and application hooks
    |-- test_evidence.py            Clean-frame files, hashes and boundaries
    |-- test_violation_export.py    Selected exports and privacy defaults
    `-- test_fastapi.py             ASGI lifecycle, broadcasting and persistence
```

The root and SORT `.gitignore` files are also tracked. Model weights, video,
virtual environments, benchmark datasets, caches, generated outputs, IDE files,
local secrets, and the local PDF report are excluded from Git.

## Environment and required local assets

Validation uses the existing **Windows CPython 3.8.10** environment, without
installing packages or modifying `myvenv`. The installed compatible web stack is
FastAPI 0.120.4, Starlette 0.44.0, Pydantic 2.10.6, Uvicorn 0.33.0 and HTTPX
0.28.1. `requirements.txt` pins the complete installed dependency closure for the
application, HTTPX tests and offline tools, including PyTorch 2.4.1+cu121,
Ultralytics 8.3.217, EasyOCR 1.7.2, OpenCV 4.12.0.88 and NumPy 1.24.1.
Flask is no longer an application dependency; no package was uninstalled.
SQLite comes from Python's standard library.

**Python 3.11/3.12 is preferred for future development.** No usable interpreter of
those versions is installed here. Python 3.8 and the pinned web stack are a
compatibility baseline, not a current production platform. Do not install the
latest FastAPI/Uvicorn into Python 3.8. A newer runtime needs a separately
approved environment and dependency validation: OpenCV 4.12 requires NumPy 2
on Python >=3.9, so this exact NumPy 1.24.1 specification cannot be reused there.
No ML upgrades or new interpreter installation were performed.

To reproduce the tested environment in a **separate** directory, after obtaining
an approved Python 3.8.10 interpreter (do not recreate or modify `myvenv`):

```powershell
& <python-3.8.10-path> -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

These are setup instructions, not commands executed during migration. The local
virtual environment is not distributed. Both OpenCV distributions in the pins
mirror the supplied environment (Ultralytics/cvzone require `opencv-python`,
EasyOCR requires `opencv-python-headless`); a fresh build must confirm the shared
`cv2` import and video/JPEG tests. Optional `util.load_model` alone needs
Streamlit, which is not part of the web application's dependency closure.

Provide these existing, separately obtained assets at the following paths:

```text
models/yolov8n.pt
models/license_plate_detector.pt
video/sample.mp4
```

Both weight files and the default video are validated before resource startup.
The application resolves these paths relative to `main.py`, independently of the
terminal working directory. It does not substitute a webcam or alternative
YOLO weights if an asset is missing.

EasyOCR also needs its English detector/recognizer weights in its external cache
(normally `~/.EasyOCR/model`). Downloads are disabled; provision that cache in advance for offline use.
No model weights or recordings are included in the repository.

## Run locally

From the project root, using the prepared Windows environment:

```powershell
.\myvenv\Scripts\python.exe -B -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1 --timeout-graceful-shutdown 5
```

Alternatively, use the equivalent Python executable from your own compatible
environment. Open `http://127.0.0.1:8000/`, watch the stream, and use the existing
red/green buttons to control the simulated signal. `python -B main.py` also runs
Uvicorn on this address. API documentation is at `/docs`, with schema at
`/openapi.json`. Do not use multiple workers or reload: each process would own a
separate capture, tracker and SQLite session. Public production hosting is not
configured by this migration.

The five-second graceful HTTP shutdown deadline lets Uvicorn cancel ongoing
streams before lifespan joins the processing owner. Keep it for webcam sources;
without a deadline, open streams can delay shutdown indefinitely.

The video does not loop automatically. EOF or a terminal processing error stops
the shared processing session; restart the process to start another session.
Other HTTP routes remain available after video completion. Lifespan initializes
resources once; the first video viewer starts consumption. Processing continues
to EOF even if every viewer disconnects. Slow viewers skip older frames rather
than advancing the source or building queues. See [Phase 7](docs/phase7_fastapi_migration.md).

Vehicle confidence defaults to the original effective `0.25`; inference size is
`640` and ROI filtering is disabled. Optional startup environment settings select
confidence, inference size, device, an existing compatible local vehicle model,
local video or `webcam:N`. Configuration is read before resource initialization;
restart to change an active session. The 720 x 480 processing and crossing
coordinates remain unchanged. See [Phase 3 detection documentation](docs/phase3_detection.md)
for settings, benchmark commands, accuracy dataset formats and limitations.

SORT retains its default `max_age=1`, `min_hits=3`, `iou_threshold=0.3`.
`TRAFFIC_SORT_MAX_AGE`, `TRAFFIC_SORT_MIN_HITS` and
`TRAFFIC_SORT_IOU_THRESHOLD` configure startup experiments. The application
suppresses native tentative startup output, never confirms crossings from pure
predictions, and retains earned confirmation for observed short-gap recovery.
See [Phase 4 tracking documentation](docs/phase4_tracking.md) for exact semantics,
tradeoffs, lifecycle metadata, headless imports and identity evaluation.

## Crossing configuration

Defaults are defined in `CrossingConfig` in `violation_logic.py` and instantiated
as `CROSSING_CONFIG` in `main.py`. Coordinates refer to the resized processing
image, not the original video resolution.

| Setting | Default |
| --- | --- |
| Processing size | 720 x 480 |
| Stop-line endpoints | (160, 320) to (550, 320) |
| Prohibited direction | Downward |
| Crossing anchor | Integer bottom-center minus 30 pixels vertically |
| Hysteresis / endpoint tolerance | 2 pixels each |
| Minimum directed movement | 2 pixels |
| Maximum observation gap | 3 frame indices |
| Maximum displacement between observations | 80 pixels |
| Position-history expiration | More than 90 inactive frames |

These are initial scene settings, not measured camera calibration. The anchor
is an image-space approximation rather than a calibrated ground-contact point.
The geometry also supports sloped lines and configurable cardinal directions.

Every frame uses one signal snapshot and version. A signal transition between
observations makes crossing timing uncertain. A crossing first observed within
the hysteresis band can be accepted later after sufficient departure movement,
using the signal and frame captured at the original intersection.

## Routes and candidate lifecycle

| Route | Response / behavior |
| --- | --- |
| `GET /` | Existing HTML interface |
| `GET /video` | Shared MJPEG with `boundary=frame`; 503 on unavailable startup |
| `POST /set_light/red` | Set simulated red, then redirect to `/` |
| `POST /set_light/green` | Set simulated green, then redirect to `/` |
| `POST /set_light/{other}` | 422 validation error; signal unchanged |
| `GET /violations` | Current-session JSON list of recognized plate strings |
| `GET /docs`, `/openapi.json` | Automatic API documentation and schema |

A red crossing creates an in-memory candidate before OCR succeeds. Pending OCR
may continue while that vehicle remains visible, including after the light turns
green. Once a plate is recognized for the candidate, repeated recognition work
for that candidate stops. OCR does not promote an event to a confirmed violation.

Plate detector/OCR score cutoffs default to 0.25/0.65; an unmodified score >=0.90
can qualify immediately, otherwise two separate-frame observations are required.
Legacy seven-character syntax remains configurable for compatibility. Geometric
ambiguities and conflicting corrections stay unresolved; provisional text never
enters `/violations`. Fixed inverse threshold 64 remains the default. See
[Phase 5 OCR documentation](docs/phase5_ocr.md) for all settings, diagnostic
counters, scheduling, format limitations, tool schemas and measured results.

The existing frontend polls `/violations` every 500 ms. This compatibility
endpoint is not an event API: it omits candidates without recognized plates and
returns remembered plate strings rather than a list of currently visible cars.
The public plate list remains process-local and does not reveal stored history.
Candidate records now persist by default in ignored `data/violations.db`, including
events without plates. Database failure retains in-memory candidates with explicit
failure diagnostics and bounded retries. Review/history are local backend operations.
Evidence capture defaults off; enable `TRAFFIC_EVIDENCE_ENABLED=true` only for an
authorized local run. See [Phase 6 storage documentation](docs/phase6_storage.md)
for configuration, privacy, failure recovery, exports and explicit retention.

## Tests

With the local default video and model files present, run:

```powershell
.\myvenv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The suite has **445 tests**: all 412 Phase 1-6 tests (only framework-specific
client/startup assertions adapted), plus 33 FastAPI/producer/concurrency tests.
New checks cover lifespan, one-owner broadcasting, slow/disconnected viewers,
loop responsiveness, current-session JSON, and single-session SQLite/OCR/evidence.
Existing application fixtures
explicitly disable production storage; new storage tests use temporary databases.
Model and OCR inference
are mocked in automated tests. Small real OpenCV
video/JPEG and SORT checks are included. Deterministic trajectories cover signal
changes, finite endpoints, direction, jitter, missing observations, history
expiration, deduplication, uncertain timing, and OCR enrichment.

Passing these tests does not establish field accuracy, plate-recognition
accuracy, or legal validity. Those require labeled footage and camera-specific
validation.

## Known limitations

- One process/one Uvicorn worker owns video processing. Viewers receive the
  latest frame, so a slow viewer can skip frames. This is a local shared-stream
  prototype, not a distributed or multiple-worker processing service. Native
  inference cannot be forcibly cancelled; shutdown reports a timeout if it hangs.
- SORT IDs may fragment or switch under occlusion. Deduplication by track ID
  cannot fully deduplicate physical vehicles across ID changes.
- Runtime candidate/crossing memory still grows with the number of recorded tracks.
  SQLite stores historical records, but storage failures cannot guarantee durability.
  There is no secondary durable outbox or automatic retention cleanup.
- OCR supports selected Indian layouts, BH-series and optional legacy syntax,
  not all legal registrations. Low-resolution, split-line and ambiguous reads
  can remain unreadable. The default fixed threshold is configurable; without
  labeled plate images no preprocessing or field-accuracy improvement is claimed.
- Source, geometry, and thresholds require per-camera configuration. A
  prerecorded video and wall-clock manual controls do not provide real-world
  signal synchronization.
- Evidence is opt-in local storage, not an atomic signed evidence bundle. Local
  OS permissions protect records; no authenticated public review interface is provided.
- Yellow-light behavior, automatic signal recognition, re-identification,
  authentication and production deployment are not implemented.
- The offline helpers are separate workflows. `add_missing_data.py` expects
  `test.csv`; `visualize.py` expects `test_interpolated.csv` and a root-level
  `sample.mp4`. The active FastAPI pipeline does not produce these CSV inputs.

## Existing demonstration images

These images are retained from earlier project documentation. They illustrate
prior demonstrations and are not measurements of the current detector's accuracy.
The recognized-plate display is in memory, not permanent storage.

![Earlier startup demonstration](https://github.com/user-attachments/assets/c32c3231-9630-49bb-b685-66b83cdbbfee)

![Flask traffic monitoring interface](https://github.com/user-attachments/assets/031d8b03-4b8e-4fcc-a6c3-534eeae095b9)

![Earlier red-light highlighting demonstration](https://github.com/user-attachments/assets/e2cbd195-3b3d-4fe0-88d5-3945d34828b8)

![License plate recognition demonstration](https://github.com/user-attachments/assets/018fb558-8142-4561-bf05-3914635b11bc)

![Recognized plate list](https://github.com/user-attachments/assets/ec8cb04c-6403-4f60-906c-54cb988a6d44)

## Project history and SORT attribution

The existing GitHub history and the current application's history are preserved
through a merge commit. The obsolete root-level `Car_Counter.py`, `sort.py`, and
`mask.png` are absent from the latest tree; they remain retrievable from commit
`da4652d540b08a921ff51ff1cac7ff5f378efa22`, for example:

```powershell
git show da4652d:Car_Counter.py
```

There is no `legacy/` copy. Removing root `sort.py` also prevents it from
shadowing the active `sort/` package.

The vendored SORT implementation retains its original attribution, documentation,
and GPL license in `sort/sort.py`, `sort/README.md`, and `sort/LICENSE`.
