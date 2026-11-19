# Phase 5: license plate detection and OCR

Phase 5 adds conservative plate syntax checks, bounded event evidence, safe
crops, unambiguous geometric vehicle association, and synchronous OCR scheduling.
It does not change the crossing algorithm, YOLO weights, SORT defaults or frontend.
Recognition acceptance remains a hypothesis attached to a violation **candidate**;
neither registration legality nor physical vehicle identity is verified.

## Implementation and compatibility

- `plate_recognition.py`: pure normalization, validation, corrections, ranking,
  image operations, association, configuration and a bounded `OCRManager`.
- `util.py`: existing helper names delegate to these rules. `read_license_plate`
  returns a structured `Reading` that still unpacks as `(text, score)`.
  Tuple-returning recognition providers are also accepted by the manager.
  CSV export and the optional Streamlit loader retain their existing interfaces.
- `main.py`: one best candidate crop per event/frame, status/counter updates,
  OCR scheduling and accepted-result enrichment. Event UUIDs are never replaced.
- `tools/benchmark_ocr.py` and `tools/evaluate_ocr.py`: independent offline tools.

`GET /`, `GET /video`, `POST /set_light/<color>` and `GET /violations` retain
their existing behavior. `/violations` remains a JSON list of accepted strings.
Provisional OCR never reaches that list or the display plate memory.

EasyOCR is lazily initialized on the first recognition request, CPU/English,
with `download_enabled=False`. Missing/corrupt cached weights cause an explicit
failure, never a download. Heavy Streamlit and Ultralytics imports in `util.py`
are local to their optional helper. Importing the pure rules needs no model.
The [EasyOCR implementation](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/easyocr.py)
documents the reader/download flag and the detailed text/confidence output.

## What the actual failure diagnosis showed

The old code tested only a seven-character `LLDDLLL` layout, accepted the first
matching OCR row without a confidence cutoff, used strict whole-plate containment
and a first-vehicle match, and ran OCR every visible unresolved frame. Raw NumPy
slicing did not validate finite coordinates, clipping or minimum dimensions.

A controlled comparison used **the original committed Phase 4 main and util**
from `2467bc1`, then Phase 5, on the first 60 consecutive existing sample-video
frames. Both created two candidates and supplied exactly identical accepted
vehicle detections to SORT. Downloads were disabled and socket connections were
blocked. Existing local YOLO and cached EasyOCR weights were used unchanged.

The baseline made 21 OCR calls: **20 returned nothing**. Its only returned text
had five characters, failed the old layout, and scored below 0.65. Thus replacing
the validator cannot by itself repair this footage. Crop readability, text
detection, preprocessing and recognition remain material problems. There are no
plate/vehicle identity labels for this video, so this is a failure-stage diagnosis,
not a measurement of missed plates, correct association or recognition accuracy.
Raw strings and crop images were not written into these diagnostic reports.

Counters on `main.ocr_manager.counts` distinguish:

- `plates_detected`, `confidence_rejected_plates`, `invalid_crops`;
- `associated_plates`, `failed_associations`, `valid_crops`;
- `ocr_attempts`, `ocr_results`, `empty_results`, `ocr_exceptions`;
- `format_rejected_results`, `low_confidence_results`,
  `invalid_confidence_results`, `ambiguous_corrections`,
  `duplicate_observations`, `accepted_results`.

Absent counter keys mean zero. Detector counters refer to returned YOLO rows;
confidence removals internal to YOLO are not observable in these counters.
Association counts include noncandidate vehicles; valid-crop counts concern
scheduled candidate events and are not necessarily actual OCR calls.
Aggregate counters are DEBUG logged at most once per 60 frames. Ordinary logs
contain no raw OCR strings. `TRAFFIC_OCR_DIAGNOSTIC=true` explicitly permits raw
EasyOCR outputs in the `util` DEBUG logger for local inspection; logging must also
be configured to DEBUG. Keep this opt-in disabled during ordinary use.

## Supported syntax, normalization and correction

Default standard syntax is a configured state/UT prefix, 1-2 district digits,
1-3 series letters and **four** serial digits. Examples:

```text
UP15AB1234   KA03MN1234   DL8CAF5032   MH12AB1234   TN1A1234
22BH1234AA  (Bharat: two digits, literal BH, four digits, two letters)
AB12CDE     (explicit backward-compatible legacy layout)
```

The supported prefix set is visible as `STATES`; it includes some historical
prefixes for compatibility. It is a deployment syntax allowlist, not an
authoritative current registration registry. Other special/historical layouts
are rejected unless configured through `OCRConfig.extra_patterns`, a tuple of
`(format_name, regex)` pairs. Custom patterns are fully matched and receive no
speculative corrections. For example:

```python
from dataclasses import replace
from plate_recognition import OCRConfig, validate_plate
config = replace(OCRConfig(), allow_legacy=False,
                 extra_patterns=(('deployment', r'TEST[0-9]{3}'),))
result = validate_plate('up-15 ab 1234', config)
# Validation(normalized_text='UP15AB1234', format_type='standard', is_valid=True)
```

Normalization removes ASCII whitespace and hyphens and uppercases ASCII text.
Unsupported punctuation is retained to fail validation; it is not erased into a
plausible registration. Nonstring, non-ASCII and over-64-character inputs yield
empty normalized output. Original accepted OCR strings are retained separately
in bounded event observations. No partial boxes/lines are concatenated into
invented registrations; split or two-line OCR may therefore remain unreadable.

Position-specific confusion pairs are **O/0, I/1, Z/2, S/5, B/8, G/6**. A
letter slot permits digit-to-letter correction and a digit slot permits the
reverse. Literal BH characters are never guessed. Already valid text is returned
unchanged. At most two edits and eight candidate strings are permitted by
default; a candidate cap overflow is rejected, never truncated to a false winner.
Variable layouts can produce multiple valid strings: `UPI5AB1234` could yield
`UP15AB1234` or `UP1SAB1234`, so it remains conflicting. Unmodified valid strings
rank ahead of speculative corrections, then by confidence and lexical tie order.
Competing results within 0.05 score at the same correction count are conflicting.
Up to 64 OCR rows are examined; duplicate strings within one call give one vote.

## Configuration

Startup validates settings before opening capture. Restart to change active
settings. Probabilities must be real, finite, nonboolean and in [0,1]. Scores are
not calibrated probabilities of exact registration correctness.

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `TRAFFIC_PLATE_CONFIDENCE` | 0.25 | Plate YOLO prefilter and application cutoff |
| `TRAFFIC_OCR_CONFIDENCE` | 0.65 | Minimum eligible OCR score |
| `TRAFFIC_OCR_SINGLE_READ_CONFIDENCE` | 0.90 | Unmodified, noncompeting single-read acceptance |
| `TRAFFIC_OCR_MIN_OBSERVATIONS` | 2 | Repeated-observation consensus fallback |
| `TRAFFIC_OCR_HISTORY_SIZE` | 12 | Maximum observations per active event |
| `TRAFFIC_OCR_HISTORY_FRAMES` | 90 | Observation age limit |
| `TRAFFIC_OCR_INTERVAL_FRAMES` | 1 | Minimum retry/plate-scan interval |
| `TRAFFIC_OCR_MAX_BACKOFF` | 8 | Maximum failed-attempt interval multiplier |
| `TRAFFIC_OCR_MAX_ATTEMPTS` | 20 | Actual OCR attempts per event |
| `TRAFFIC_OCR_STATE_TTL` | 150 | Frames without visible event before state expiry |
| `TRAFFIC_OCR_MAX_STATES` | 128 | Concurrent event evidence limit |
| `TRAFFIC_OCR_MIN_WIDTH` / `MIN_HEIGHT` | 16 / 8 | Clipped crop dimensions in processing pixels |
| `TRAFFIC_OCR_MAX_CORRECTIONS` | 2 | Maximum substitutions (supported range 0-3) |
| `TRAFFIC_OCR_MAX_CANDIDATES` | 8 | Maximum interpretations per OCR result |
| `TRAFFIC_OCR_PREPROCESSING` | fixed | color, gray, fixed, otsu, adaptive, contrast |
| `TRAFFIC_OCR_UPSCALE` | 1 | Bounded scale, 1-3 |
| `TRAFFIC_OCR_ALLOW_LEGACY` | true | Retain seven-character compatibility |
| `TRAFFIC_OCR_DIAGNOSTIC` | false | Permit raw results in explicit local DEBUG logs |

`single_read_confidence` must be >= `ocr_confidence`; observation count cannot
exceed history size. Set the single-read threshold to 1 to reserve single-read
acceptance for engine score 1; this is a separate shortcut from minimum votes.
Corrected strings always require repeated evidence under the default minimum.
The defaults retain compatibility; local labeled validation is needed to choose
thresholds for a deployment. For Indian-only operation disable legacy syntax.

## Crops and association

Only finite real positive-area boxes can reach cropping. Bounds clip to the
720 x 480 clean inference frame; floor/ceil produce enclosing pixel bounds.
Reversed boxes are not swapped. Empty/outside/tiny crops are rejected before
OpenCV preprocessing. Crops and transforms are independent copies.

The compatibility `get_car` now requires the plate center inside a vehicle,
at least 70% of plate area contained, and plate area <=40% of vehicle area.
The score is `0.8*contained_fraction + 0.2*min(relative_area/0.1, 1)`.
The best candidate must beat the next by at least 0.12; otherwise association
is unassigned. Boundary-touching and modest partial containment are allowed.
Duplicate IDs, invalid vehicles, nearby but unsupported matches and ties are
rejected. OCR text never selects a vehicle. This geometry is a heuristic, not
an association-confidence probability; wrong confident matches remain possible.

All confirmed observed vehicles participate in association, even if only one
has a candidate event, avoiding attribution simply because another vehicle is
not awaiting OCR. Each event receives only its highest area*detector-score crop
per frame. Multiple plate detections in one frame cannot manufacture consensus.

## Evidence, scheduling and event safety

Evidence is keyed by event UUID and never shared between IDs. Each bounded
observation stores frame, original and normalized string, OCR/detector scores,
correction count and recognition status. Rejected attempts store their status;
raw rejected strings are inspected only through explicit diagnostic mode.

Eligible identical strings receive confidence-weighted votes:
`OCR_score * detector_score / (1 + 0.2*corrections)`. A candidate requires the
configured number of separate-frame observations and a >1.5x weight advantage
over the next string. A sole uncorrected read >=0.90 can qualify immediately,
preserving historical single-frame behavior. Lexical order makes reporting
deterministic, but tied evidence is never accepted. The strongest observed OCR
score is retained, not an invented combined probability. Better provisional
observations can improve selection; weaker reads cannot overwrite an accepted
event. Existing `enrich_event` still rejects weaker replacements and preserves
event UUID/status. After accepted evidence, no more OCR is scheduled.

Recognition statuses (`not_attempted`, `pending`, `unreadable`, `low_confidence`,
`format_unrecognized`, `conflicting`, `accepted`) live in `event['recognition']`,
separate from candidate status. Provisional text is internal. Raw observation
history is discarded on acceptance, expiry or shutdown; bounded summary metadata
stays on the already persistent candidate. Candidate/plate records themselves
retain the existing session lifetime and may grow with event count as before.

Failed-attempt intervals grow linearly up to 8x the minimum. A crop whose
area*detector score grows by at least 50% can retry at the minimum interval.
Plate scans still run at the minimum interval to notice a clearer crop; OCR
backoff does not claim a reduction in plate-model calls. Attempt limits count
actual OCR calls, not misses, rejected associations or invalid crops. There is
no thread, job queue or background state.

Invalid/ambiguous tracking output, expired identities, missing prior history,
or a post-event observation gap/jump beyond the unchanged crossing safety limits
close further OCR for that event and clear observations. The event is retained.
Short gaps within those limits can recover; numeric-ID continuity is not proof
of physical identity, and plausible identity switches remain undetectable.
OCR enrichment continues after red becomes green. Disappearance and recognition
failures never delete candidates or create duplicate events.

Phase 1's terminal shared-stream policy remains: malformed nonfinite/reversed/
degenerate plate geometry or an OCR/model exception stops once and releases
capture, with stage/status diagnostics and candidate preservation. Small/clipped
unusable crops are skipped safely. No invalid crop reaches OpenCV. This phase
does not introduce retry loops after a potentially inconsistent shared resource
failure or redesign multi-client streaming.

## Preprocessing and measurements

The default remains grayscale + threshold 64, binary inverse. Alternatives are
the clean BGR crop, gray, Otsu binary threshold, adaptive Gaussian threshold
(block 11, C=2), and grayscale CLAHE (clip 2, 8x8 tiles). Optional cubic scaling
is limited to 1-3x. Only one selected strategy runs per production attempt.

Final controlled smoke (CPU AMD64 Family 23 Model 24, 8 logical CPUs, 7 torch
threads; Python 3.8.10, torch 2.4.1+cu121, Ultralytics 8.3.217, OpenCV 4.12.0,
NumPy 1.24.1; default nano .25/640/.7 and SORT 1/3/.3):

| Measurement, first 60 frames | Phase 4 baseline | Phase 5 |
| --- | ---: | ---: |
| Plate-model calls / returned detections | 21 / 95 | 21 / 95 |
| Associated detections, all vehicles | 89 | 76 |
| Unassigned detections | 6 | 19 |
| Valid candidate crops | 21 | 21 |
| Actual OCR calls | 21 | 6 |
| OCR calls / processed frame | 0.35 | 0.10 |
| OCR calls / candidate | 10.5 | 3.0 |
| Mean / median / P95 OCR helper ms | 74.73 / 57.83 / 164.26 | 57.24 / 56.62 / 69.22 |
| Total OCR helper time, seconds | 1.569 | 0.343 |
| Full processing seconds / FPS | 23.177 / 2.589 | 19.385 / 3.095 |
| Candidate events / enriched events | 2 / 0 | 2 / 0 |
| Accepted strings | 0 | 0 |

OCR calls fell 71.4%; total OCR helper time fell in this run. These per-call
latencies involve different scheduled crops and are not a same-image algorithm
speedup. Full FPS includes decode, inference, crossing, drawing and MJPEG
generation, excludes startup/warmup/HTTP/browser. Sequential timing varied: an
earlier comparison measured 3.184 vs 3.104 FPS. Therefore **no reliable overall
throughput improvement is claimed**. Both runs preserve all candidates and
recognize zero plates; this does not demonstrate increased accuracy.

Six identical unlabeled clean candidate crops were also compared across all six
preprocessing strategies, and color/fixed at controlled 2x scale. No accepted
plate was found in the original-scale six-method comparison. Some transforms
returned more noise; that is not improvement. Reports contain exact sample
hashes, timings, counts and configuration. Without labels there is no basis to
choose a new default. Generated, ignored reports:

```text
outputs/phase5_application_smoke_final.json
outputs/phase5_preprocessing_smoke_final.json
outputs/phase5_preprocessing_upscale_smoke.json
```

## Optional offline tools and annotation schema

Prepare local clean plate images and a manifest; labels must be independently
read/verified, never inferred from OCR. Paths are relative to the manifest.
Empty text denotes a deliberately labeled negative sample. Format labels are
provided independently, not inferred from the recognizer.

```json
{
  "schema_version": 1,
  "samples": [
    {"id": "plate-001", "image": "plate-001.png", "text": "UP15AB1234", "format": "standard"},
    {"id": "negative-001", "image": "negative-001.png", "text": "", "format": "negative"}
  ]
}
```

Omit `text`/`format` for an explicitly unlabeled latency-only manifest. Tools
never load Flask, download weights, install packages or modify input images.

```powershell
.\myvenv\Scripts\python.exe -B tools/benchmark_ocr.py `
  --manifest datasets/ocr/manifest.json --methods color gray fixed otsu adaptive contrast `
  --repeats 3 --warmup 1 --output outputs/ocr_benchmark_new.json

.\myvenv\Scripts\python.exe -B tools/benchmark_ocr.py `
  --manifest datasets/ocr/manifest.json --methods color fixed --upscale 2 `
  --output outputs/ocr_upscale_new.json

.\myvenv\Scripts\python.exe -B tools/evaluate_ocr.py `
  --annotations datasets/ocr/manifest.json --predictions outputs/ocr_benchmark_new.json `
  --output outputs/ocr_metrics_new.csv
```

Benchmark predictions are `samples` with matching IDs and a `strategies` object:

```json
{"id":"plate-001", "strategies":{"fixed":{"text":"UP15AB1234","accepted":true}}}
```

Here `accepted` means **single-crop format/confidence eligibility**, not production
event consensus. Benchmark timings include mean/median/P95 engine call latency,
preprocessing overhead and crop-pipeline FPS. Decode/init/warmup and the full
video pipeline are excluded. Each method processes identical images; predictions
come from the first measured replay, timing/counts from all measured repeats.
The optional `--processed-frames` and `--violation-candidates` denominators must
refer to the recorded sequence represented by the manifest; rates use one
sequence's denominators per replay and remain null if unknown.

Evaluation requires exactly the same unique IDs and complete strategy coverage.
Metrics are exact accepted-plate accuracy over positive samples; micro CER using
Levenshtein edits / true characters (rejections count as empty predictions);
acceptance/rejection over all samples; wrong accepted strings / all accepted
strings (`false_acceptance_rate`), and wrong accepted / all samples separately.
Negative samples can expose false acceptances. Undefined denominators are null.
Per-format and per-strategy breakdowns use the same definitions. JSON/CSV exports
record provenance hashes and refuse overwrites. Reports are restricted to ignored
`outputs/`. Do not compare crop acceptance against production consensus acceptance.

Explicit independently labeled `plate_detected`/`correct_association` booleans in
saved predictions can report stage-observation rates for positive examples.
Missing/incomplete stage labels yield null. These coarse rates do not infer
detector recall from an OCR crop dataset, nor constitute a full association
benchmark. Proper detector recall needs all plate boxes, association accuracy
needs matched vehicle identities, and end-to-end exact-event recognition needs
matched video event truth. No such data was supplied; those metrics are unmeasured.

## Verification and limits

```powershell
.\myvenv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The 219 existing tests remain unchanged. New checks cover example/legacy/custom
formats, punctuation, nonstring inputs, position-aware and ambiguous corrections,
confidence ranking, multiple/empty outputs, bounded histories, separate-frame
votes, ties/conflicts, independent events, best-result preservation, safe crop
boundaries, every preprocessing option, overlapping vehicles, clean frames,
scheduling/limits, red-to-green enrichment, unchanged JSON API and failure cleanup.
Tool tests use known synthetic labels and mocked readers, not real model files.
The complete suite passes **319 tests: 219 unchanged existing tests plus 100 new
tests** (78 recognition/application and 22 tool tests), with zero failures.
All seven changed/new Python files compile under Python 3.8.10. Fresh imports
initialize no capture, weights, EasyOCR or optional Streamlit. A SHA-256 audit
confirms only `main.py`, `util.py` and `README.md` changed among original files;
all original assets/tests, SORT licensing, tracking/detection/crossing modules
and old reports are unchanged. All 46,834 virtual-environment files retain the
same size/mtime metadata digest. No original files were deleted.

No labeled real plate set was provided, so **exact-match accuracy, CER, false
acceptance, detector recall, association accuracy and end-to-end accuracy are not
measured on the real smoke**. The six diagnostic crops are a tiny correlated
sample, not a validation set. GPU/webcam, adverse weather, moving cameras, split
two-line plates, many historical/special formats, and live GUI viewing remain
untested. Fixed thresholding may hurt some plates; scoring/geometry heuristics
need labeled tuning. SORT identity switches can still cause attribution errors.
Accepted strings require review and are not authoritative evidence.

No new dependency, model, external API, owner lookup, training, database, queue,
frontend redesign or Phase 6 functionality is introduced. Phase 5 is left
unstaged/uncommitted on `main`; no branch, history rewrite or push is performed.
