# Phase 4 SORT tracking reliability

Phase 4 keeps YOLO, SORT, the 720 x 480 processing coordinates, the Phase 2
crossing algorithm, OCR rules, Flask routes, and frontend intact. It adds a
small lifecycle adapter and offline tracking tools. No model/recording/dataset
or virtual environment is changed, and no additional dependency is installed.

## Parameters and exact vendored semantics

Startup environment variables are optional and read once before opening the
capture. Restart the process to change them; active streams cannot reset or
switch trackers through HTTP.

| Environment variable | Default | Validation |
| --- | ---: | --- |
| `TRAFFIC_SORT_MAX_AGE` | 1 | Integer >= 0 |
| `TRAFFIC_SORT_MIN_HITS` | 3 | Integer >= 1 |
| `TRAFFIC_SORT_IOU_THRESHOLD` | 0.3 | Finite number in [0,1] |

`max_age` is the number of consecutive missed updates retained. Removal occurs
when `time_since_update > max_age`, after prediction and matching. At the default,
one empty frame is retained; a detection on the next update can recover the same
track. Two consecutive empty updates expire it. A new detection after expiration
creates a new ID. Calling `update` once for **every frame**, including empty
frames, is essential. There is no new automatic extrapolated-output mode.

`min_hits` in this source is a **consecutive successful association count**:
creating a track initializes `hits=hit_streak=0`. Three associated updates mean
four observed frames before strict first confirmation. Native SORT also emits
observed tentative tracks during its first `min_hits` global updates; the
application adapter suppresses this startup bypass for crossing safety.

After a miss, native SORT resets `hit_streak` on the next prediction, and normally
suppresses a recovered track until its streak reaches `min_hits` again. The
adapter preserves an already earned confirmation and can emit the **observed,
measurement-corrected** state immediately when that same retained track recovers.
It does not create an association of its own or resurrect an expired track.
This makes brief-gap observation handling explicit; it does not prove physical
identity. Newly created tracks still need full confirmation.

`iou_threshold` gates native association. This implementation uses `>` in the
unique-association fast path and rejects `< threshold` Hungarian matches, so
exact equality can depend on the branch taken. That upstream policy is unchanged.
Zero/very low thresholds can admit incorrect associations, including disjoint
boxes in the Hungarian branch. Longer retention can also increase incorrect
associations. The existing defaults remain unchanged; experiments are not
recommendations:

```powershell
$env:TRAFFIC_SORT_MAX_AGE = '3'
$env:TRAFFIC_SORT_MIN_HITS = '3'
$env:TRAFFIC_SORT_IOU_THRESHOLD = '0.3'
.\myvenv\Scripts\python.exe -B main.py
```

Invalid settings fail startup with a diagnostic error before capture/model
loading; `/video` returns 503. All Phase 3 environment settings remain available.

## Lifecycle and observed versus predicted positions

`TrackingManager` owns one native `Sort` instance. SORT still owns the Kalman
filter and detection association. The adapter derives first/last frame numbers
and missed counts from native age fields rather than keeping trajectory copies.
Immutable `TrackSnapshot` records are exposed through `mot_tracker.tracks`:

| State | Metadata | Sent to crossing/OCR/rendering |
| --- | --- | --- |
| Tentative | Current observed estimate, not yet confirmed | No |
| Confirmed | Confirmed and updated by a detection this frame | Yes |
| Lost | Retained prediction, `observed=False` | No |
| Recovered | Observed update of a retained track; one-update notification | Yes if previously confirmed |
| Expired | Removed from live snapshots; one-update notification | No |

Each snapshot records tracking ID, first-seen frame, last-observed frame,
consecutive misses, confirmation, clipped bbox, observation flag, and state.
An observed bbox is a Kalman posterior corrected by the current detection,
not an unfiltered detection box. Predicted-only boxes are visible as metadata
but never establish a crossing, request OCR, or get drawn as observed vehicles.
Pure predictions may cross the line without creating an event.

Only active-track snapshots and confirmation flags are retained. Recovered and
expired notifications cover one update; creation/expiration/recovery totals are
integer counters. EOF or terminal shutdown clears live tracking state without
resetting IDs or deleting events. Debug logging reports active/recovered/expired/
invalid tracks. No tracking REST API or frontend change was introduced.

Track input is checked for numeric `(N,5)` shape, finite coordinates, valid
confidence and positive clipped area before native state advances. Output is
validated as `[x1,y1,x2,y2,positive_integer_id]`. Nonfinite/degenerate boxes are
excluded. Every occurrence of a duplicated output ID is excluded rather than
selecting an arbitrary vehicle. Rejected known IDs invalidate only transient
crossing history, preserving recorded crossings and OCR results.

## Identity and event safety

No appearance network, extra reassociation, proximity merge, cross-ID deduplication,
or OCR-based identity merge is introduced. Native SORT IDs remain process-local,
monotonically allocated identifiers; they are not physical vehicle identities.
Occlusion, overlap and reentry can still fragment or switch them.

The existing crossing detector still requires a preceding approach observation,
finite line intersection, permitted direction, hysteresis, acceptable frame gap/
jump, and consistent signal state/version. A newly confirmed ID first seen beyond
the line cannot create a violation merely by appearing there. Retaining a track
for five misses does not enlarge Phase 2's default maximum observation gap of
three frame indices. Pure predictions cannot bridge that limit.

Candidates, recorded crossing IDs, event UUIDs and plate memory remain separate
from live tracks. Expiration, invalid-output cleanup and shutdown do not erase
them. Existing per-ID duplicate prevention remains. Two distinct vehicles, even
with identical OCR text or nearby crossings, retain separate events. An expired
vehicle receiving a new ID and later making a separately observed crossing can
still create another candidate; there is no reliable physical-identity evidence
to suppress it automatically. SORT can also associate a different vehicle to an
existing ID under ambiguity. These limitations require labeled evaluation.

New strict confirmation can miss short-lived crossings during startup or before
a new track confirms; that is an intentional conservative tradeoff. Persisting
event/plate records still grow over a session as before; the adapter adds no
unbounded history or cross-ID cache.

## Vendored changes, dependencies and attribution

The original Alex Bewley copyright and GPL-3.0-or-later notice remain in
`sort/sort.py`. `sort/LICENSE` is unchanged. A dated local modification notice and
`sort/README.md` describe all upstream changes:

1. Plotting/Tk-related imports move into the `--display` demonstration. Tracking
   imports without Matplotlib, scikit-image or Tk. The demo honours the selected
   Matplotlib backend rather than forcing `TkAgg`.
2. Demo colors use a local seeded RNG; importing tracking does not reset the
   application's NumPy RNG.
3. Nonfinite/nonpositive-area predictions and their tracker objects are removed
   together. Previously an infinite prediction could be masked from the matrix
   without removing its tracker, shifting association indices onto another ID.

Kalman matrices/equations, association strategy and native defaults are unchanged.
Core tracking uses existing NumPy, FilterPy, and SciPy (or optional installed LAP).
The historical SORT requirements file still includes optional demo dependencies.
No replacement tracker code or new licensing terms were introduced.

## Recorded detections and parameter benchmarking

Use the same saved detector output for every experiment. Input is JSON:

```json
{
  "schema_version": 1,
  "frame_size": [720, 480],
  "metadata": {"model": "yolov8n.pt", "confidence": 0.25, "imgsz": 640, "nms_iou": 0.7},
  "frames": [
    {"frame_index": 0, "detections": [[100, 100, 180, 200, 0.9]]},
    {"frame_index": 1, "detections": []}
  ]
}
```

This example illustrates the schema, not measured footage. Include every frame
in order from zero, including empty frames. Each row is validated YOLO-to-SORT
`[x1,y1,x2,y2,confidence]`. Record model/source hashes, actual inference settings,
video frame sampling and preprocessing in metadata for reproducibility. The
benchmark does not load models, open video, use Flask resources or mutate input.

```powershell
.\myvenv\Scripts\python.exe -B tools/benchmark_tracking.py `
  --detections outputs/phase4_recorded_detections.json `
  --configs 1,3,0.3 3,3,0.3 5,3,0.3 3,2,0.3 `
  --repeats 3 --warmup-replays 1 --compare-native --include-tracks `
  --output outputs/tracking_comparison_new.json
```

Each replay creates an independent tracker without resetting/reusing native ID
counters. Numeric IDs can differ between replays; association counts should not.
Mean/median/P95 `update` latency, tracker-only FPS, active-track mean/peak/final,
creations, expirations and emitted observations are reported. Warmup replays are
excluded. Counts refer to one replay, timings pool measured replays. Hardware,
NumPy/SciPy versions, assignment backend, detector metadata and input hashes are
recorded. The optional native comparison uses native SORT output, including its
startup bypass; adapter output has the safety gates described above. Native
timing covers `Sort.update`; adapter timing includes its validation/lifecycle work.

These FPS figures exclude YOLO, decoding, crossing, OCR, rendering, JPEG and
Flask. They are not application FPS. Track creation/expiration and ID counts are
**not** ID-switch accuracy metrics. Retaining more tracks is not proof of better
identity tracking. Reports refuse to overwrite existing files and should remain
under ignored `outputs/`.

With `--include-tracks`, each result embeds a compatible `predictions` document.
Extract the desired result's `predictions` to a new JSON file for the evaluator.
Use `--annotations labels.json` to include class-agnostic identity metrics when
complete identities for exactly that sequence are genuinely known.

## Identity evaluation and annotation preparation

No matching identity-labeled ground truth was supplied for the application video.
Real ID switches, fragmentation, IDF1, MOTA and identity precision/recall remain
unmeasured. Existing SORT benchmark recordings/detections are not assumed to be
vehicle identities or annotations of `video/sample.mp4`.

Prepare a small contiguous validation clip including separate vehicles, overlap,
occlusion, exits and reentries. Label every visible supported vehicle on every
frame with a persistent physical identity; keep the same ID across known gaps.
Use consistent 720 x 480 processing coordinates. Fully label negative/empty
frames too. Never infer truth IDs from SORT's output. Keep ambiguous/incomplete
intervals out of the selected clip; the utility requires a fully annotated,
contiguous sequence and refuses incomplete truth. Do not modify original media.
Store prepared labels separately, e.g. under ignored `datasets/`.

Annotation format:

```json
{
  "schema_version": 1,
  "frame_size": [720, 480],
  "metadata": {"data_kind": "manually_annotated", "source": "clip identification"},
  "frames": [
    {"frame_index": 0, "fully_annotated": true,
     "objects": [{"id": 1, "class_id": 2, "bbox": [100, 100, 180, 200]}]},
    {"frame_index": 1, "fully_annotated": true, "objects": []}
  ]
}
```

Predictions use the same schema, size, ordered frame coverage and `objects`,
without the `fully_annotated` requirement. Include model/tracker parameters and
output policy in prediction metadata. IDs must be positive integers and unique
within each frame. Boxes must be finite, positive-area and in bounds; labels are
never clipped or rewritten. Missing frames/duplicate IDs fail validation.

```powershell
.\myvenv\Scripts\python.exe -B tools/evaluate_tracking.py `
  --annotations datasets/tracking/labels.json --predictions outputs/tracks.json `
  --iou 0.5 --output outputs/tracking_metrics_new.json
```

JSON or CSV exports include input hashes, provenance/configuration and metrics.
`--class-aware` additionally gates matches by COCO vehicle class (2,3,5,7) and
requires classes on all objects. Native SORT does not retain class IDs, so its
generated replay tracks are evaluated class-agnostically, explicitly reported.
No extra package is installed; the utility uses the existing SciPy assignment.

Metric conventions (not an official MOTChallenge submission pipeline):

- Local matching prioritizes eligible previous-frame matches, then Hungarian IoU
  with IoU >= the cutoff. IDs are sorted for repeatable ties. One match per object.
- ID switches compare a GT identity's current predicted ID to its last matched
  ID, including across gaps. Fragmentation is matched -> missed labeled
  appearance(s) -> matched. Absence from GT is not a missed labeled appearance.
- False-positive/negative appearances are unmatched predictions/labels per frame.
  False-positive tracks are predicted IDs never locally matched; missed tracks
  are GT identities never locally matched. These are different counts.
- MOTA = 1 - (FN + FP + ID switches) / GT appearances; it can be negative.
- Identity counts use a **global one-to-one** GT/predicted identity assignment
  maximizing eligible spatial pair counts over all frames, not local-match votes.
  ID precision = IDTP / predicted appearances; ID recall = IDTP / GT appearances;
  IDF1 = 2*IDTP / (GT appearances + predicted appearances).
- Per-ID continuity is matched / labeled appearances; mean continuity averages
  those fractions. Undefined denominators are null. No metrics are invented for
  footage without identity labels.

The conventions were checked against the primary
[TrackEval CLEAR implementation](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py)
and [identity implementation](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py).
The local implementation is independent and omits official benchmark-specific
ignored-region/distractor preprocessing. Do not compare its numbers directly to
published MOTChallenge results without matching dataset conventions.

## Regression tests and remaining limits

```powershell
.\myvenv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The 154 previous tests remain unchanged. New tests use synthetic detections and
mocked application inference/OCR; they do not require model weights. Coverage
includes exact retention/confirmation semantics, empty updates, short/long gaps,
recovery, expiry/reentry, crossing/overlapping trajectories, duplicate/invalid
outputs, bounded metadata, no predicted crossing, persistent event/plate memory,
two nearby distinct events, startup failures, import without plotting/Tk, the
infinite-prediction indexing regression, replay counts/timing units, and
hand-calculated global identity/fragmentation/MOTA fixtures.

The complete suite passes **219 tests: 154 existing plus 65 new** (39 tracker/
application checks and 26 tracking-tool checks), with zero failures. Syntax was
checked against Python 3.8. A fresh subprocess blocked Matplotlib, scikit-image
and Tk imports and successfully imported Flask, SORT and both offline tools and
ran tracking. No additional dependency or plotting initialization was required.

Real physical identity through occlusion remains uncertain. Longer retention,
more observations, fewer emitted IDs or fewer creations do not establish an
accuracy improvement. Live webcam/GPU and an interactive plotting window were
not validated. The single shared capture/viewer design, manual signal timing,
session-local event memory and existing OCR limitations remain unchanged.

## Measured validation results

Hardware/software: Windows development machine, AMD64 Family 23 Model 24 CPU,
8 logical processors, NumPy 1.24.1 and the supplied SciPy/FilterPy environment.
Full inference used CPU, 7 PyTorch threads, Python 3.8.10, Ultralytics 8.3.217,
PyTorch 2.4.1+cu121 and OpenCV 4.12.0. No GPU, download or package install was
used. The JSON reports retain hardware, source/model hashes and configuration.

`outputs/phase4_recorded_detections.json` contains the exact same 1,304 accepted
YOLO detections from the first 60 consecutive `video/sample.mp4` frames. Vehicle
settings were nano, confidence 0.25, imgsz 640, NMS IoU 0.7, ROI off, processing
size 720 x 480. Each replay comparison used these identical detections, 3 measured
replays and 1 excluded warmup replay; tracker IoU was 0.3 in every row.

| Tracker / max_age / min_hits | Mean ms | Median ms | P95 ms | Tracker-only FPS | Created / expired | Active final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native SORT / 1 / 3 | 6.133 | 6.066 | 7.392 | 163.1 | 71 / 48 | 23 |
| Adapter default / 1 / 3 | 9.341 | 9.317 | 11.284 | 107.1 | 71 / 48 | 23 |
| Adapter experiment / 3 / 3 | 9.777 | 9.731 | 11.728 | 102.3 | 63 / 39 | 24 |
| Adapter experiment / 5 / 3 | 9.813 | 9.857 | 11.354 | 101.9 | 61 / 36 | 25 |
| Adapter experiment / 3 / 2 | 9.870 | 9.838 | 11.062 | 101.3 | 63 / 39 | 24 |

Default active mean/peak was 22.93/28; age 3 gave 24.48/29 and age 5 gave
25.75/29. The adapter adds validation and lifecycle overhead; no speed gain is
claimed. Larger ages reduced creations in this clip, but that is not measured
identity accuracy and might hide wrong associations. Defaults remain 1/3/0.3.
Native experimental rows and emitted-observation counts are in
`outputs/phase4_tracking_benchmark.json`. Fewer emitted IDs may reflect tentative
track suppression, not physical identity preservation. Real IDSW/IDF1/MOTA
remain unmeasured because the recorded clip lacks identity labels.

A separate instrumented single-viewer smoke compared baseline
`ebc2b7c:main.py` (native SORT) against the updated application, on the same first
60 frames, manual red, CPU/default model settings and 3 warmup calls per YOLO
model. Both used the finite-prediction fix, which leaves valid Kalman math
unchanged. Actual detector inputs were compared exactly and matched.

| Measurement | Phase 3 native | Phase 4 adapter |
| --- | ---: | ---: |
| Full per-frame processing FPS | 2.809 | 2.655 |
| Mean vehicle model-call latency | 194.58 ms | 211.44 ms |
| Mean tracker-update latency | 10.87 ms | 16.55 ms |
| Emitted observed track rows | 1,182 | 1,120 |
| Unique emitted IDs | 47 | 44 |
| Crossing candidates | 2 | 2 |
| Plate-model calls, all on clean frames | 21 | 21 |
| Actual OCR-function calls | 23 | 21 |
| Recognized plates | 0 | 0 |

The full interval includes decoding, inference, tracking, crossing, any triggered
plate/OCR work, drawing and MJPEG generation, but excludes initialization,
warmup, HTTP delivery and browser rendering. Sequential runs varied in model
latency; no application throughput improvement is claimed. The Phase 4 tracker
created 71 tracks, expired 48, recovered 22 and retained 23 at frame 59; shutdown
then released capture and live tracking state. Routes `/`, `/violations`, and
both red/green controls worked. All returned tracker arrays were finite, `(N,5)`
and had unique IDs. OCR used existing cached weights with downloads explicitly
disabled. See `outputs/phase4_application_smoke.json` for full timing statistics.

Finally, `outputs/phase4_synthetic_tracking_benchmark.json` exercises the identity
evaluator against **known synthetic truth**, not real footage. One downward
trajectory spans 20 frames with detections missing at frames 8 and 9:

| Adapter max_age / min_hits | ID switches | Fragmentations | IDF1 | MOTA |
| --- | ---: | ---: | ---: | ---: |
| 1 / 3 | 1 | 1 | 0.4375 | 0.55 |
| 3 / 3 | 0 | 1 | 0.8571 | 0.75 |
| 5 / 3 | 0 | 1 | 0.8571 | 0.75 |
| 3 / 2 | 0 | 1 | 0.8889 | 0.80 |

This controlled example verifies retention/metric behavior for an unambiguous
gap. It does not justify changing defaults or claim an improvement on real
overlapping vehicles. Missing observations still cause fragmentation because
predictions are not emitted as observations. Synthetic inputs and labels are
explicitly marked as such and stored under ignored `outputs/`, separately from
all original datasets. Existing benchmark reports were not overwritten.
