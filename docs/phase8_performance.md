# Phase 8: performance profile and shared-stream optimization

Baseline: `337cffa3e703b21e71d80eeb87f8adfff01d124a` on `main`.
The working tree was clean and all 445 Phase 1-7 tests passed before editing.
No dependencies, interpreter, weights, recordings, datasets or virtual
environment files were installed, upgraded or modified. Changes remain
unstaged; no branch, commit or push is part of this phase.

## Changes and scope

Modified: `main.py`, `processing_service.py`, `web_api.py`,
`evidence_manager.py`, `README.md`, and one synchronization condition in
`tests/test_fastapi.py`. All assertions in that existing test remain; it now
waits explicitly for the producer's next-step barrier, because notification
delivery can precede that step rather than waiting for a polling interval.

Added: `performance_monitor.py`, `pipeline_profiling.py`,
`tools/benchmark_pipeline.py`, `tests/test_performance.py`,
`tests/test_pipeline_benchmark.py`, and this document.

The CV modules, SORT, crossing geometry, signal versioning, OCR rules and
scheduler, SQLite schema/retries/review/export, frontend files and attribution
remain unchanged. Evidence encoding/writing has timing contexts only, retaining
the original paths, JPEG quality, exclusive writes, flush/fsync, hash validation
and failure handling. There is no background evidence/database writer.

## Reproducible measurement

`tools/benchmark_pipeline.py` uses a fresh subprocess for each trial, sequentially.
No two inference trials compete. Fixed references read existing local Git objects:
`phase7` loads main/web/service from `337cffa`; `flask` loads `8626582` with its
matching committed template. No checkouts or Git mutations occur. `current`
uses the working source and explicitly controlled profiling/pacing. The
historical Flask package is still installed locally, so both versions were
executed. It is not restored as an active application dependency.

Common workload: first 60 consecutive `video/sample.mp4` frames, red signal,
720 x 480 processing, original vehicle/plate models, vehicle confidence 0.25,
configured inference size 640, original SORT/OCR settings, SQLite enabled,
evidence disabled except the explicit evidence scenario. Every model has three
warmup calls and EasyOCR is warmed on a blank image. Warmup consumes a separate
probe frame, never the capture used by tracking; measured indices remain 0-59.
Initialization, model loading and warmup are recorded separately, outside steady
state FPS. No confidence/format/consensus rule or inference opportunity is reduced.

Throughput experiments are explicitly **unpaced**; normal playback is separate.
A capture wrapper ends each test source at frame 60 without editing or seeking
the recording. No frame sampling or skipped detector input is used. Temporary
SQLite/evidence live in unique ignored `outputs/pipeline-trial-*` directories.
Connections to the network are blocked during inference, with cached models and
OCR only. Windows' local event-loop socketpair is created before the block.

Processing FPS counts encoded completed frames from first acquisition through
last JPEG. Streaming FPS counts delivered frames over that same interval.
Model-only throughput is the inverse of model call latency, not pipeline FPS.
Acquisition means immediately before the native capture read. Delivery means
in-process ASGI send; neither camera backend queueing nor network/browser/display
latency is included. The comparison keeps hardware/device/settings equivalent.

Reports include per-frame validated detections and confirmed observed tracks,
with track IDs normalized by first appearance, plus crossing frame/point/source
time/signal/status. UUIDs and wall-clock UTC times intentionally differ across
fresh sessions; persisted UUIDs are checked against their live events within a
trial. The comparator requires identical structures, frame indices, identities
and normalized event attributes; detection/track floats allow only absolute
1e-4 tolerance with no relative tolerance. Actual accepted plates are zero in
this workload, so no recognition-text equivalence or accuracy claim is made.
Reports contain no plate strings. Candidates/persistence are checked beyond
counts; evidence references are rehashed when enabled.

## Baseline and bottlenecks

The repeated CPU Phase 7 baseline measured **3.117 median FPS**, range
**3.112-3.286**, across three trials. The Flask reference measured **2.998 median
FPS**, range **2.808-3.013**. The earlier isolated 1.94-FPS Phase 7 result was not
reproduced in these baseline trials; it is not evidence that FastAPI caused a
regression. Later samples also vary substantially, as reported below.

The initial profiled trial attributed about 56% of frame time to vehicle YOLO,
18% to plate YOLO, and 14% to capture. Across the three Phase 7 baseline trials:

| Stage | Calls in 60-frame run | Median of trial mean latencies | Median exclusive time |
| --- | ---: | ---: | ---: |
| Vehicle YOLO | 60 | 183.95 ms | 11.037 s |
| Plate YOLO | 21 | 164.14 ms | 3.447 s |
| Capture | 61 reads, 60 valid frames | 42.83 ms/read | 2.613 s |
| EasyOCR | 6 | 53.08 ms | 0.318 s |
| Stream JPEG | 60 | 2.73 ms | 0.164 s |

Five SQLite dispatches averaged about 6.53 ms each in the median trial, far
below inference time. Full database operations contain transactions and must
not be added to their nested transaction timings. Similarly, detection includes
inference, tracking lifecycle includes SORT, OCR includes EasyOCR, and pipeline
contains its stages. Timing attribution uses exclusive/self time. Delivery
latency overlaps pipeline execution and is excluded from contributor ranking;
publication, lock waits and pacing have separate scopes.

## Retained optimizations

The measured subscriber bottleneck was Phase 7's 10-ms polling, often quantized
to roughly 15 ms on this Windows/Python runtime. It did not explain all processing
time, but added avoidable delivery delay. Polling is replaced with an asyncio
event per viewer, notified using the viewer loop's thread-safe callback API.
Only one callback can be queued per subscriber; a queued or already-signaled
notification coalesces further publications. Slow viewers read the latest slot
and cannot build a frame or callback history. Events are set on their own loop.
Disconnects remove their waiter; closed loops do not crash the producer.

The existing one-slot immutable JPEG buffer remains. Each display frame is
encoded once and the same bytes go to all subscribers. No inference lock is
held while notifying clients or awaiting transport. A new viewer receives the
latest frame without counting publications from before it connected as skips.
An explicit stop is checked before requesting the next generator frame,
avoiding an extra unnecessary inference after shutdown was already requested.

Three one-viewer notification trials measured mean encode-to-ASGI delays
**0.747, 0.765 and 0.805 ms**, versus baseline **8.924, 8.617 and 7.666 ms**:
about **91% less mean delivery delay** in this comparison. Detections, tracks
and normalized crossings matched. Notification-trial median processing FPS was
3.289 (range 2.892-3.298), about 5.5% above the earlier CPU baseline median;
the ranges overlap, so this is **not a reliable general FPS speedup claim**.

## Tuning evaluated and rejected

An attempted two-thread CPU setting initially did not take effect: installed
Ultralytics `select_device` resets PyTorch threads to seven while setting up CPU
prediction. Effective budgets are now verified by the benchmark on the actual
processing owner. Three effective two-thread trials measured 1.768, 2.194 and
2.674 FPS, slower than defaults; strict trace equivalence also failed. The
application CPU-thread override/wrapper was removed. `--cpu-threads` remains
only a clearly labeled isolated benchmark experiment, set after model warmup.

The installed predictor already uses `smart_inference_mode`; another inference
mode wrapper would duplicate working behavior. Existing model reuse and clean
frame separation are retained. CUDA timings use installed Ultralytics' existing
synchronizing profiler; no half precision or new model/backend is enabled.

Plate inference already runs only for eligible events. The same 21 plate calls
and six OCR attempts remain. No crop cache or duplicate-image rejection was
added: an unproven change could remove a useful OCR opportunity or alter
multi-frame consensus. There were 1,304 validated vehicle detections, two
candidates and zero accepted plates in the reference workload. OCR accuracy is
not improved by this phase.

Frame copies preserve clean inference/crop/evidence input while annotations use
their separate image. Their measured cost did not justify aliasing or removing
that safety. JPEG and storage are small contributors, so their defaults,
transaction durability and synchronous persistence-before-OCR remain unchanged.

## Repeated comparisons and additional scenarios

| Scenario | Trials | Median processing FPS | Range |
| --- | ---: | ---: | --- |
| Historical Flask, CPU, profiled | 3 | 2.998 | 2.808-3.013 |
| Phase 7, CPU, profiled | 3 | 3.117 | 3.112-3.286 |
| Current, CPU, profiled | 3 | 3.289 | 2.892-3.298 |
| Current, CPU, profiling disabled | 3 | 2.333 | 2.250-2.366 |
| Two subscribers, CPU, profiled | 1 | 1.885 | exploratory |
| Evidence enabled, CPU, profiled | 1 | 1.944 | exploratory |
| JPEG 80, CPU, profiled | 1 | 1.582 | exploratory |
| JPEG 90, CPU, profiled | 1 | 2.591 | exploratory |
| Two CPU threads, rejected | 3 | 2.194 | 1.768-2.674 |
| Phase 7, GTX 1650, profiled | 3 | 6.972 | 5.349-7.347 |
| Current, GTX 1650, profiled | 3 | 8.122 | 5.370-8.169 |

The observed CUDA median difference is +16.5%, but ranges overlap and samples
remain small. Across GPU versions, full traces and normalized events matched.
CPU and GPU must not be treated as identical configurations: their maximum
box differences were about 0.000259 pixels and track differences about 0.000121
pixels; crossing attributes remained equal. This is not a new GPU/half-precision
optimization: existing device selection is retained, and the GPU scenario is
explicitly labeled. No field accuracy inference follows from these tiny deltas.

The rejected two-thread experiment also retained crossing attributes, but its
maximum detection difference of 0.000122 pixels exceeded the deliberately strict
1e-4 trace gate. It was not made a default or retained as application tuning.

Two final interleaved CPU pairs, Phase 7 then current, measured 2.139 -> 2.838
and 2.741 -> 2.620 FPS: one faster and one slower. Their mean delivery delays
were 8.329 -> 0.960 and 8.399 -> 0.930 ms, respectively. The consistent result
is lower subscriber delay, not a consistent processing-FPS improvement. A
well-controlled host and longer runs are needed to attribute throughput changes.
The original Phase 7 slowdown's specific cause cannot be established retroactively.

The disabled-profiling samples were taken later under different local load and
were slower than the earlier profiled samples. This does not imply profiling
accelerates inference or provide a causal overhead estimate. An isolated
five-repeat, 5,000-call no-op microbenchmark measured 0.065 microseconds per
original disabled call and 10.264 microseconds per enabled timed call (median).
It excludes ML, resources and transport; it is a wrapper-cost measurement only.
Disabled instrumentation returns the original callable, with no timing clock.

All non-rejected CPU scenarios matched reference detections, observed tracks and
normalized event attributes. Both real-model viewers received all 60 frames,
with one set of 60 encodes, six OCR attempts and two persisted events. Their mean
delivery delays were 1.230 and 1.448 ms; no duplicate source/session/inference was
created. Intentional detector-frame skips stayed zero. Slow viewers may skip
publications, as the existing latest-frame design intends.

Evidence enabled produced ten saved, hash-verified files for two events.
Measured means in that run: evidence capture 8.95 ms, evidence JPEG 2.54 ms,
exclusive file writing/fsync 1.75 ms, event insertion 9.55 ms and evidence
metadata 7.72 ms. These scopes nest; they must not be added as independent work.
Every required image retained its requested/source-frame metadata and integrity
checks. Storage remains synchronous and before OCR.

Mean display JPEG sizes were 135,704 bytes at quality 95, 93,137 at 90 and
62,118 at 80: roughly 31% and 54% less data in these experiments. Detection/OCR
input and evidence quality did not change. These single-trial quality runs do
not establish an FPS benefit; lower display quality can reduce human readability.
Quality 95 remains the default.

## CPU, memory, startup and resource findings

Hardware: Windows build 26200, AMD Family 23 Model 24, eight logical CPUs,
approximately 5.88 GiB usable system RAM, GTX 1650 with 4 GiB VRAM. Runtime:
Python 3.8.10, Torch 2.4.1+cu121, Ultralytics 8.3.217, OpenCV 4.12.0 and
NumPy 1.24.1. CPU prediction used the library's seven-thread budget. CUDA
prediction retained the library's existing device behavior. Both configurations
were run on the same machine; no competing inference experiments were launched.

The initial CPU baseline had median process CPU 586% (about 5.86 logical cores),
median end-of-trial RSS 444 MiB and peak working sets up to 1,054 MiB. Current
profiled CPU trials had median RSS 473 MiB and a similar maximum peak of
1,054 MiB. No memory reduction is claimed. Windows working-set trimming and
system load varied; available memory was often only about 0.5-1.3 GiB, and
significant timing variation also appeared in initialization and inference.
This is a plausible confound, not proof of the historic slowdown's cause.

The first GPU scenario measured 160% process CPU, peak working set 1,234 MiB,
about 24 MiB allocated/68 MiB reserved by Torch and 3.14 GiB GPU memory free in
sampled snapshots. Allocator memory excludes CUDA driver/context overhead.
GPU utilization percent was not measured. All figures are workload-specific;
end RSS is not a substitute for peak memory.

In the first CPU baseline, native resource startup was about 0.954 s, model
loads 59/48 ms, cached EasyOCR reader initialization 5.428 s, and warmup
2.576 s. Other trials varied (reader initialization roughly 5-8 s). Native startup
does not include earlier Python/ML package imports; newer reports separately
record worker-function-entry to pre-warmup readiness, excluding interpreter
launch. Production retains lazy EasyOCR, so its first real OCR attempt can incur
reader initialization that the warmed steady-state benchmark intentionally
excludes. No cold-start delay is disguised as a throughput improvement.

The profile uses bounded scalar metadata rather than retaining video frames.
The stream still holds one latest JPEG; an active client may retain one in-flight
part under backpressure. Notification callbacks coalesce and OCR histories remain
bounded as before. Crossing candidate/event memory can still grow with the number
of recorded tracks; that earlier limitation is not a claimed memory optimization.

Full ignored artifacts: `outputs/phase8_phase7_baseline.json`,
`phase8_flask_baseline.json`, `phase8_notifications.json`,
`phase8_without_profiling.json`, the two-viewer/evidence/JPEG/CUDA scenario JSONs,
interleaved trial JSONs, `phase8_profiling_overhead.json`, and
`phase8_summary.json`. CSV trial summaries are also supported; JSON contains the
full nested stage and correctness data. Earlier `phase8_threads2.json` records
the ineffective requested budget and is not used as a valid two-thread result;
`phase8_threads2_effective.json` is the actual rejected experiment.

## Configuration, pacing and private observability

| Setting | Default | Validation |
| --- | --- | --- |
| `TRAFFIC_PROFILE` | false | true/false/1/0 |
| `TRAFFIC_PROFILE_EVERY` | 1 | integer 1-1000 |
| `TRAFFIC_PROFILE_WINDOW` | 512 | integer 16-4096 |
| `TRAFFIC_MONITOR_SECONDS` | 1 | finite 0.1-60 seconds |
| `TRAFFIC_JPEG_QUALITY` | 95 | integer 1-100, stream only |
| `TRAFFIC_PACING` | source | source or unpaced |

No new settings package is used. Existing vehicle device/imgsz, ROI, tracking,
plate and OCR settings/defaults remain in their existing configuration classes.
Invalid performance settings fail before opening capture. Stream quality 95
preserves the original OpenCV default call; other qualities are explicit opt-in.
Evidence JPEG stays at 95 regardless of stream quality.

Source pacing is an intentional playback correction: a fast prerecorded
pipeline waits outside locks until its source timestamp deadline. Frame order,
indices and timestamps are not changed. Source FPS supplies a fallback for
unavailable timestamps; missing/invalid FPS falls back to unpaced with unknown
source rate. Slow processing never skips detector frames to catch up. Stop
events interrupt pacing waits. Benchmark mode explicitly uses unpaced timing.

For live sources, no unbounded application capture queue is introduced, but
backend/camera buffering is outside this prototype's control. Slow processing
can accumulate source latency; it is reported rather than hidden with frame
dropping. Local code/debugger can inspect
`main.app.state.processing.snapshot()` for active viewers, publications, latest
buffer depth, queued callback count, throughput/source FPS and whether it keeps
up. There is no public diagnostic endpoint or raw observation payload.

Profiling uses `perf_counter_ns`, fixed stage names and per-thread nesting.
Disabled instrumentation installs no proxy wrappers and reads no profiling
clock. Enabled data includes lifetime timed-call counts/totals/maxima and bounded
recent median/p95 samples; a sampling interval greater than one means these
timing totals are sampled, not extrapolated full-session costs. Event/frame
counters remain complete. Frame samples are bounded by the configured window;
resource snapshots by min(window,120). No images, plates, UUIDs or event histories
are stored by the profiler. No per-frame logs are emitted.

CPU percent and RSS use existing psutil; CPU percent can exceed 100 on a
multicore process. Peak working set includes startup/warmup. GPU allocated,
reserved, peak and available memory are sampled only after CUDA is already
initialized; profiling never initializes CUDA just to observe it. GPU utilization
is not collected through an additional package or polling subprocess.

## Running experiments

```powershell
.\myvenv\Scripts\python.exe -B tools/benchmark_pipeline.py --reference phase7 --frames 60 --warmup 3 --trials 3 --profile --output outputs/baseline.json
.\myvenv\Scripts\python.exe -B tools/benchmark_pipeline.py --reference current --frames 60 --warmup 3 --trials 3 --profile --output outputs/current.json --csv outputs/current.csv
# Independent scenarios, never launched concurrently:
.\myvenv\Scripts\python.exe -B tools/benchmark_pipeline.py --viewers 2 --profile --output outputs/two_viewers.json
.\myvenv\Scripts\python.exe -B tools/benchmark_pipeline.py --evidence --profile --output outputs/evidence.json
.\myvenv\Scripts\python.exe -B tools/benchmark_pipeline.py --jpeg-quality 80 --profile --output outputs/jpeg80.json
```

Omit `--profile` for the disabled-profiling comparison. `--device cpu` is the
explicit comparison default, `--device 0` requires an available CUDA GPU, and
`--device auto` preserves automatic device selection. Unavailable requested
devices fail rather than silently falling back. New JSON/optional CSV paths
must resolve inside ignored `outputs/`; existing files and paths outside that
root are rejected. Benchmark traces are bounded by the maximum 6,000 measured
frames and retain geometry metadata, never full video images.

## Verification and limitations

The test command remains:

```powershell
.\myvenv\Scripts\python.exe -B -m unittest discover -s tests -v
```

New tests cover configuration bounds, disabled clocks/wrappers, nested timing,
exceptions, sampling/windows, thread-safe aggregates, privacy, optional resource
metrics, source/slow/unknown-FPS pacing and stop interruption, single JPEG
encoding, clean inference, persistence-before-OCR, evidence timing/integrity,
coalesced callback bounds, disconnects/closed loops, no polling, accurate skip
counters, no extra inference after stop, benchmark output/device validation and
strict per-frame/event comparisons. All prior regression tests are retained.

Final result: **502 tests passed, zero failures**: all 445 previous tests plus
57 focused new checks. Syntax/import and dependency checks also pass. Live
Uvicorn/HTTPX tests (mocked inference, actual transport/JPEG) verified two
viewers sharing one producer, one disconnecting while the other continued,
homepage/static/docs/signal/JSON routes, and no surviving threads. An open-ended
stream shutdown test with a shortened 0.2-s graceful deadline stopped server and
producer in about 0.37 s with zero remaining viewers. The real launcher still
uses its existing five-second HTTP deadline and 30-second native-worker limit.
These are separate from the real-model in-process ASGI timing. No production
database was created.

Final protected-file audit: only the six intended existing files changed; no
baseline file disappeared. Model/video/dataset/frontend and other original
source hashes remain unchanged. All 46,834 virtual-environment files retained
their original count, sizes and modification times. The index is empty and
all 12 Phase 8 source/documentation/test changes remain unstaged on `main` at
`337cffa`. The branch's existing one-commit lead over `origin/main` is the Phase 7
commit, not a new Phase 8 commit. No commit, push, branch or remote change occurred.

No labeled accuracy dataset, long-running camera, browser/network latency load
test or high-resolution GPU-utilization measurement is claimed. Python 3.8 and
the existing ML/runtime versions are unchanged. Candidate memory and native-call
shutdown limits from earlier phases remain; this is not a distributed or
multi-worker processing service.

Primary sources checked: installed Ultralytics 8.3.217 predictor/torch_utils/ops
source, [PyTorch CPU thread API](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html),
and [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/).
