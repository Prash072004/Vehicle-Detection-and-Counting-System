# Phase 3 detection configuration and validation

The application still uses `models/yolov8n.pt`, `video/sample.mp4`, a 720 x 480
processing image, the existing SORT parameters, and the Phase 2 crossing rules.
The plate model, OCR preprocessing/recognition, frontend, endpoints and MJPEG
format are unchanged. Vehicle configuration is independent of plate inference.

## Startup configuration

Set environment variables in the server's shell **before** starting `main.py`.
All are optional. Relative paths resolve against the project root.

| Variable | Default | Accepted values |
| --- | --- | --- |
| `TRAFFIC_VEHICLE_CONFIDENCE` | `0.25` | Finite number from 0 to 1 |
| `TRAFFIC_VEHICLE_IMGSZ` | `640` | Positive integer multiple of 32 |
| `TRAFFIC_VEHICLE_DEVICE` | Backend automatic selection | `cpu`, `mps`, or a GPU index such as `0` |
| `TRAFFIC_VEHICLE_MODEL` | `models/yolov8n.pt` | Existing compatible local COCO detection weights |
| `TRAFFIC_VIDEO_SOURCE` | `video/sample.mp4` | Existing local video path or `webcam:0` (nonnegative camera index) |
| `TRAFFIC_ROI_ENABLED` | `false` | `true` / `false` / `1` / `0` |
| `TRAFFIC_ROI` | Unset | `x1,y1,x2,y2` in 720 x 480 processing coordinates |
| `TRAFFIC_ROI_ANCHOR` | `bottom_center` | `bottom_center` or `center` |

For example, these are **experimental settings**, not measured recommendations:

```powershell
$env:TRAFFIC_VIDEO_SOURCE = 'video/sample2.mp4'
$env:TRAFFIC_VEHICLE_CONFIDENCE = '0.30'
$env:TRAFFIC_VEHICLE_IMGSZ = '960'
$env:TRAFFIC_VEHICLE_DEVICE = 'cpu'
# Enable only after calibrating your own scene. This example is illustrative.
$env:TRAFFIC_ROI = '100,80,650,470'
$env:TRAFFIC_ROI_ENABLED = 'true'
.\myvenv\Scripts\python.exe -B main.py
```

Unset these variables or use a fresh shell to restore defaults. Configuration
errors fail initialization with a diagnostic message; `/video` returns HTTP 503.
Missing files are rejected before opening a capture or loading models. Webcam
availability errors release the attempted capture. No fallback model or source
is substituted. Remote/network sources and source selection by web users are
unsupported. Configuration does not switch an active session: restart the
process to change sources or settings. EOF/read failures remain terminal.

## Detection filtering and image ownership

`vehicle_detector.py` accepts only car (2), motorcycle (3), bus (5) and truck (7)
rows. Before SORT it rejects malformed/non-numeric/nonfinite rows, invalid
confidence, fractional class IDs, and zero/reversed areas. Coordinates clip to
the current image edges; boxes with no area after clipping are rejected. SORT
always receives a floating-point `(N, 5)` array, including `(0, 5)` when empty.

YOLO receives the configured confidence cutoff before NMS. One defensive
application boundary validates its returned rows; there is no second application
NMS/filtering pass. That boundary accepts confidence equal to the cutoff. The
installed Ultralytics 8.3.217 backend uses a strict `>` cutoff internally; an
exactly equal candidate may therefore never be returned. Passing `conf=0.25`
preserves that existing effective backend behavior. Counts of rejected confidence
values cover returned rows only, not candidates removed inside YOLO.
Vehicle NMS IoU remains the installed backend's default `0.7`, now passed
explicitly and recorded in tool reports. It is distinct from evaluation matching
IoU (`0.5` by default). Plate inference arguments are unchanged.

ROI filtering is disabled by default. Enabled rectangles require positive area,
finite bounds inside their reference image, and a matching processing size.
The representative point of the **clipped** box must be inside the rectangle;
rectangle boundaries count as inside. A box extending beyond an ROI is accepted
when its representative point is inside. Partial overlap alone is insufficient.
The default point is bottom-center. This setting is separate from the Phase 2
crossing anchor (which retains its existing vertical offset). Polygon ROIs are
not implemented in this phase.

Each resized frame has one clean inference array and one display copy. Both
YOLO models and OCR crops use the clean array. Lines, rectangles, event colors,
plate text, and JPEG encoding use the display copy. Drawing cannot contaminate
later inference or OCR on the same frame. No extra crop copies are introduced.

## Small and distant vehicles

Ultralytics letterboxes the processing image for `imgsz` and returns boxes in
that input image's coordinates. The application does **not** scale these boxes
again. Changing `imgsz` keeps the 720 x 480 tracking/crossing coordinate system.
See the [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/).

The existing resize to 720 x 480 can remove distant-object detail and distort
nonmatching source aspect ratios. Raising `imgsz` only resamples that processing
image; it cannot recover detail already discarded. Native-resolution inference
with a separate mapping layer is outside this change. Raising `imgsz`, lowering
confidence, selecting a local larger COCO model, or choosing an ROI are controlled
experiments: compare labeled precision/recall and speed before changing defaults.
ROI filtering follows inference and does not save YOLO computation. A larger
model or more returned detections is not evidence of improved accuracy.

## Speed benchmarking

The standalone utility checks all model files before loading any, uses its own
capture, disables downloads/automatic package installation, and writes a **new**
JSON report. It never touches Flask's capture, tracker, or OCR. Compatible local
PyTorch COCO `.pt` detectors are supported; missing/incompatible models fail.

```powershell
.\myvenv\Scripts\python.exe -B tools/benchmark_detection.py `
  --models models/yolov8n.pt models/yolov8l.pt --video video/sample.mp4 `
  --frames 20 --stride 15 --warmup 3 --conf 0.25 --imgsz 640 --device cpu `
  --compare-baseline --output outputs/phase3_benchmark.json
```

`--stride 15` samples frames 0,15,...,285 for 20 samples. Warmup is excluded from
timed samples. The report records frame indices, source/model SHA-256, model
initialization seconds, mean/median/P95 model-call milliseconds, model-call FPS,
detection-pipeline FPS, YOLO preprocessing/inference/postprocessing means,
filter counts, actual device, CPU/GPU identification, thread count, library
versions, and inference settings. Initialization includes Python library imports
for the first loaded model; it is not a fair cross-model cold-start comparison.

`--compare-baseline` compares the old vehicle adapter with the validated adapter
using the same model object, frames, settings and warmup. It requires ROI off.
It measures vehicle detection only. Neither FPS figure includes video decoding,
resizing, SORT, crossing, plate inference, OCR, rendering, JPEG or Flask. Those
figures must not be labeled complete application FPS. GPU timings synchronize
before/after model calls. Runs on this development machine are small samples,
not production throughput guarantees. Do not run CPU workloads concurrently
when comparing timings. Reports refuse to overwrite existing files.

For a higher-resolution experiment, repeat with `--imgsz 960` and a different
output filename; keep the same models, video, samples, device and warmup.

## Accuracy evaluation dataset

No compatible fully annotated vehicle validation set was supplied. Precision,
recall and F1 for the real footage are therefore **unmeasured**. Existing SORT
benchmark data is not assumed to be COCO vehicle ground truth. Unlabeled footage
can provide speed measurements and qualitative inspection only.

Prepare a small independent validation set with near/distant, occluded and
empty frames. Save frames at the processing resolution for production parity.
Annotate every visible instance of the four supported classes, including small
ones. Do not use the detector's own output as ground truth. Store data outside
tracked source, for example under the ignored `datasets/vehicle_validation/`.
Keep the original recordings and dataset files unchanged.

The format is a simple manually annotated JSON manifest, **not COCO JSON**:

```json
{
  "schema_version": 1,
  "images": [
    {
      "id": "frame000",
      "file": "images/frame000.jpg",
      "width": 720,
      "height": 480,
      "fully_annotated": true,
      "boxes": [{"class_id": 2, "bbox": [100, 150, 200, 300]}]
    }
  ]
}
```

The example box illustrates the schema and is not a real annotation. `bbox` is
absolute `[x1,y1,x2,y2]` in the saved image. Image paths are relative to the
manifest. Labels must be finite, positive-area and in bounds; they are never
clipped or rewritten. Use a unique string ID per image. An explicit empty
`boxes: []` with `fully_annotated: true` declares no supported vehicles present.
Mark incomplete images `fully_annotated: false`: the evaluator excludes them
entirely, so unlabeled vehicles do not silently become false positives. It fails
if no fully annotated images exist. Do not mark partially labeled scenes true.

```powershell
.\myvenv\Scripts\python.exe -B tools/evaluate_detection.py `
  --annotations datasets/vehicle_validation/annotations.json `
  --model models/yolov8n.pt --conf 0.25 --iou 0.5 --imgsz 640 --device cpu `
  --output outputs/vehicle_metrics.json --save-predictions outputs/vehicle_predictions.json
```

The evaluator does not resize labeled images before model inference. It checks
their dimensions against the manifest; YOLO boxes are mapped back to those
images. Use 720 x 480 saved images when comparing to the current application.
ROI is disabled during evaluation so the full labeled image is measured.

Alternatively, evaluate already saved predictions without loading a model:

```json
{
  "schema_version": 1,
  "model": {"name": "yolov8n.pt", "version": "actual generating library version"},
  "images": [
    {"id": "frame000", "boxes": [
      {"class_id": 2, "bbox": [100, 150, 200, 300], "confidence": 0.9}
    ]}
  ]
}
```

```powershell
.\myvenv\Scripts\python.exe -B tools/evaluate_detection.py `
  --annotations datasets/vehicle_validation/annotations.json `
  --predictions outputs/vehicle_predictions.json --conf 0.25 --iou 0.5 `
  --output outputs/vehicle_metrics.csv
```

Every evaluated image requires a prediction entry (empty list for no detections).
Duplicate/unknown IDs fail. Missing entries are errors, not silently empty
detections. Preserve model/library version and the generating configuration.
Lowering an evaluation confidence cutoff cannot restore predictions already
discarded when the saved file was generated. Regenerate predictions at that
cutoff or lower for a valid threshold comparison.

Matching is greedy in descending confidence, same class, one detection per label,
with IoU >= the configured threshold. Ties preserve input order. A duplicate
detection is a false positive; a wrong-class detection produces a false positive
and leaves the correct-class label unmatched. Reports contain TP/FP/FN,
micro precision/recall/F1 and per-class metrics, model/configuration provenance,
input hashes and excluded incomplete-image IDs. Undefined denominators are
`null` in JSON / empty CSV cells. JSON and CSV reports never overwrite inputs or
existing reports. This is a single operating point, not mAP; it does not measure
tracking, red-light crossing, OCR, or legal-violation accuracy.

## Regression coverage

Run the full suite with:

```powershell
.\myvenv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The original 34 stability and 52 crossing tests remain. One existing test double
now accepts inference keyword arguments; all its original assertions remain.
The complete suite currently passes 154 tests: 86 existing plus 68 new tests
(38 detection/application and 30 offline-tool checks).
New tests cover confidence limits, malformed/finite/area checks, clipped boxes,
SORT array shape/content, ROI geometry, image ownership, inference arguments,
alternate file/webcam configuration, startup failures and unchanged active
sources. Offline-tool tests cover IoU matching, duplicates, wrong classes,
missing/incomplete labels, undefined metrics, deterministic ties, JSON/CSV
exports, dataset immutability, missing models, and capture cleanup.

Live webcam/GPU hardware and field accuracy require separate validation. The
shared single-capture viewer design, ID switching, simulated signal timing,
and existing OCR format restrictions remain unchanged.

## Measured local validation

Measured during Phase 3 on this Windows development machine: AMD64 Family 23
Model 24 CPU, 8 logical processors, 7 PyTorch threads, CPU inference, Python
3.8.10, Ultralytics 8.3.217, PyTorch 2.4.1+cu121, OpenCV 4.12.0 and NumPy
1.24.1. No GPU was used. Existing weights and recordings were read unchanged;
no packages or weights were downloaded. All automated tests passed: **154/154**.

The table uses `outputs/phase3_benchmark_isolated.json` for imgsz 640 and
`outputs/phase3_benchmark_960.json` for imgsz 960. Each model/adapter used the
same 20 frames (0,15,...,285) from `video/sample.mp4`, 720 x 480 processing,
confidence 0.25, backend NMS IoU 0.7, ROI off, CPU, and 3 warmup frames.
These are model-call latencies and **detection-only** pipeline FPS.

| Model / imgsz | Adapter | Mean ms | Median ms | P95 ms | Detection-only FPS | Vehicle detections |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n / 640 | Original | 298.60 | 278.27 | 412.63 | 3.35 | 462 |
| YOLOv8n / 640 | Validated | 211.24 | 212.41 | 244.93 | 4.72 | 462 |
| YOLOv8l / 640 | Original | 2083.18 | 1952.85 | 2730.05 | 0.48 | 500 |
| YOLOv8l / 640 | Validated | 1630.33 | 1518.16 | 2127.57 | 0.61 | 500 |
| YOLOv8n / 960 | Original | 387.31 | 352.38 | 536.14 | 2.58 | 596 |
| YOLOv8n / 960 | Validated | 302.22 | 296.72 | 333.21 | 3.30 | 596 |

Initialization was 5.98 seconds for nano and 0.31 seconds for large in the
640 run; the first includes library import overhead. Nano initialization was
4.53 seconds in the separate 960 process. These are not comparable cold-start
model initialization measurements. JSON reports also contain YOLO's separate
preprocessing/inference/postprocessing timing means and the hardware/settings.
All measured returned-row rejection counts for invalid boxes, below-confidence
scores and ROI exclusion were zero (ROI was off). At 640, unsupported-class
rows numbered 2 for nano and 35 for large; at 960, nano returned none.
Unit tests separately verify rejection and ROI counts with known inputs.

An earlier repeated 640 run (`outputs/phase3_benchmark.json`) measured nano
means of 156.01/146.41 ms and large means of 1363.74/1413.63 ms for original/
validated adapters, with identical detection counts. This substantial timing
variation, and the sequential pass order, mean the table does **not** establish
a speed improvement or a stable cross-resolution/model speed ratio. More
detections do not establish better precision/recall. No accuracy is claimed.

Separate instrumented full-pipeline smoke tests used the first 60 consecutive
frames, manual red, nano/640, the same CPU/settings, and 3 warmup calls for each
YOLO model. The unchanged `d893cfb:main.py` was compared with the Phase 3 version.
OCR used existing cached English weights with downloads explicitly disabled.
The measured interval includes decoding, resize, YOLO, SORT, crossing, any
triggered plate/OCR work, drawing and MJPEG chunk generation. It excludes model
initialization, warmup, HTTP transport and browser rendering.

| Run order | Pipeline | Mean vehicle ms | Median ms | P95 ms | Full processing FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| Original then updated | Original | 181.15 | 157.78 | 304.63 | 3.07 |
| Original then updated | Phase 3 | 217.60 | 193.00 | 326.77 | 2.40 |
| Updated then original | Phase 3 | 249.73 | 242.04 | 304.50 | 2.35 |
| Updated then original | Original | 271.78 | 250.74 | 383.18 | 1.94 |

Across both runs, 120 frames divided by total processing time gives 2.381 FPS
for original and 2.376 FPS for updated (both round to 2.38). There is no
demonstrated throughput gain. Each 60-frame run produced 1,304 accepted vehicle
detections, two crossing candidates and 21 plate-model calls. All 21 updated
plate calls in each run received clean frames; original calls received drawn
annotations. No plates were recognized in these short samples. `/`,
`/violations` and red/green routes remained functional, and captures were
released. The raw reports are `outputs/phase3_application_smoke.json` and
`outputs/phase3_application_smoke_reverse.json`.

The reports are ignored local artifacts, not committed application data. Real
precision/recall/F1 remain unmeasured because no suitable labels were supplied.
