import atexit
import logging
import os
from pathlib import Path
from threading import Lock, RLock

from flask import Flask, render_template, Response, redirect, url_for, jsonify
import cv2
import cvzone
import numpy as np
from violation_logic import CrossingConfig, CrossingDetector
from vehicle_detector import DetectionConfig, detect_vehicles, project_path, resolve_video_source
from tracking_manager import TrackingConfig, TrackingManager, sanitize_tracks


app = Flask(__name__)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_ROOT / "video" / "sample.mp4"
_active_video_source = VIDEO_PATH
VEHICLE_MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
PLATE_MODEL_PATH = PROJECT_ROOT / "models" / "license_plate_detector.pt"
MAX_ENCODING_FAILURES = 3
CROSSING_CONFIG = CrossingConfig()
VEHICLE_DETECTION_CONFIG = DetectionConfig()
TRACKING_CONFIG = TrackingConfig()
crossing_detector = CrossingDetector(CROSSING_CONFIG)

# Resources belong to the process, not to an individual streaming request.
cap = None
coco_model = None
license_plate_detector_model = None
mot_tracker = None
get_car = None
read_license_plate = None
_resource_lock = RLock()
_initialized = False
_processing_finished = False
_initialization_error = None
_frame_index = -1
_last_video_timestamp_ms = None
_signal_lock = Lock()
_signal_version = 0

traffic_light = "green"

plate_memory = {}

# Geometry is calibrated in the resized processing image.
limits = [int(value) for value in CROSSING_CONFIG.line_start + CROSSING_CONFIG.line_end]


def _require_file(path, description):
    if not path.is_file():
        raise FileNotFoundError("{} is missing or is not a file: {}".format(description, path))


def _release_capture(capture):
    if capture is not None:
        try:
            capture.release()
        except Exception:
            logger.exception("Could not release video capture")


def _load_yolo_model(path):
    _require_file(path, "YOLO weight file")
    try:
        from ultralytics import YOLO
        return YOLO(str(path))
    except Exception as error:
        raise RuntimeError("Could not load YOLO model {}: {}".format(path, error)) from error


def initialize_resources():
    """Initialize once per process; terminal completion requires a process restart."""
    global cap, coco_model, license_plate_detector_model, mot_tracker
    global get_car, read_license_plate, _initialized, _initialization_error
    global crossing_detector, _frame_index, _last_video_timestamp_ms
    global VEHICLE_DETECTION_CONFIG
    global _active_video_source
    global TRACKING_CONFIG

    with _resource_lock:
        if _initialization_error is not None:
            raise RuntimeError(_initialization_error)
        if _initialized or _processing_finished:
            return

        capture = None
        try:
            detection_config = DetectionConfig.from_environment(VEHICLE_DETECTION_CONFIG)
            tracking_config = TrackingConfig.from_environment(TRACKING_CONFIG)
            detection_config.roi.validate_frame(CROSSING_CONFIG.frame_size)
            source = resolve_video_source(os.environ.get('TRAFFIC_VIDEO_SOURCE', VIDEO_PATH), PROJECT_ROOT)
            vehicle_path = project_path(os.environ.get('TRAFFIC_VEHICLE_MODEL', VEHICLE_MODEL_PATH), PROJECT_ROOT)
            if isinstance(source, Path):
                _require_file(source, "Video source")
            _require_file(vehicle_path, "Vehicle YOLO weight file")
            _require_file(PLATE_MODEL_PATH, "License plate YOLO weight file")
            try:
                capture = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
                if not capture.isOpened():
                    raise RuntimeError("VideoCapture did not open")
            except Exception as error:
                raise RuntimeError(
                    "Could not open video {}. Check file readability, corruption, "
                    "and codec support (or webcam availability): {}".format(source, error)
                ) from error

            vehicle_model = _load_yolo_model(vehicle_path)
            plate_model = _load_yolo_model(PLATE_MODEL_PATH)
            from util import get_car as associate_plate, read_license_plate as recognize_plate
            tracker = TrackingManager(tracking_config, CROSSING_CONFIG.frame_size)
        except Exception as error:
            _release_capture(capture)
            _initialization_error = "Video initialization failed: {}".format(error)
            raise RuntimeError(_initialization_error) from error

        cap = capture
        coco_model = vehicle_model
        license_plate_detector_model = plate_model
        mot_tracker = tracker
        VEHICLE_DETECTION_CONFIG = detection_config
        TRACKING_CONFIG = tracking_config
        _active_video_source = source
        get_car = associate_plate
        read_license_plate = recognize_plate
        crossing_detector = CrossingDetector(CROSSING_CONFIG)
        _frame_index = -1
        _last_video_timestamp_ms = None
        _initialized = True
        logger.info("Video resources initialized for %s; vehicle settings: %s", source, detection_config)
        logger.info("SORT tracking settings: %s", tracking_config)


def shutdown_resources():
    """Stop the shared source at EOF, a terminal error, or process shutdown."""
    global cap, _processing_finished
    with _resource_lock:
        _processing_finished = True
        capture, cap = cap, None
        _release_capture(capture)
        if isinstance(mot_tracker, TrackingManager):
            mot_tracker.clear()


def _at_video_end():
    # Container metadata is not reliable for every codec; unknown reads stop too.
    try:
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        position = cap.get(cv2.CAP_PROP_POS_FRAMES)
        return np.isfinite(count) and count > 0 and position >= count
    except Exception:
        logger.debug("Could not inspect video position", exc_info=True)
        return False


atexit.register(shutdown_resources)


def _video_timestamp_ms():
    """Use only finite, advancing container times; otherwise report unknown."""
    global _last_video_timestamp_ms
    try:
        timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        if not np.isfinite(timestamp) or timestamp < 0:
            return None
        if _last_video_timestamp_ms is not None and timestamp <= _last_video_timestamp_ms:
            return None
        _last_video_timestamp_ms = timestamp
        return timestamp
    except (cv2.error, TypeError, ValueError, OverflowError):
        return None


def generate_frames():
    global _frame_index
    try:
        initialize_resources()
    except Exception:
        logger.exception("Cannot start video stream")
        return

    # Clients still share one source and tracker. Serialize a frame, but do not
    # hold the lock across yield or release the capture on client disconnect.
    encoding_failures = 0
    while True:
        with _resource_lock:
            if _processing_finished or cap is None:
                return
            frame_nmr = _frame_index + 1
            stage = "frame reading"
            try:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    if not ret and _at_video_end():
                        logger.info("Video processing completed: %s", _active_video_source)
                    else:
                        logger.warning(
                            "Video read failed or returned an empty frame at frame %d; "
                            "stopping without retrying an unknown source position",
                            frame_nmr,
                        )
                    shutdown_resources()
                    return
                _frame_index = frame_nmr
                timestamp_ms = _video_timestamp_ms()
                # Snapshot before inference. Route changes affect later frames;
                # versions expose red/green/red changes between observations.
                with _signal_lock:
                    frame_signal, signal_version = traffic_light, _signal_version
                stage = "frame resizing"
                inference_frame = cv2.resize(frame, CROSSING_CONFIG.frame_size)
                frame = inference_frame.copy()
                stage = "vehicle inference"
                detections_array = detect_vehicles(coco_model, inference_frame, VEHICLE_DETECTION_CONFIG)

                stage = "frame annotation"
                # Draw line
                color = {"green": (0, 255, 0), "red": (0, 0, 255)}.get(frame_signal, (128, 128, 128))
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), color=color, thickness=2)

                # track vehicles
                stage = "SORT tracking"
                track_ids, invalid_ids = sanitize_tracks(
                    mot_tracker.update(detections_array), CROSSING_CONFIG.frame_size)
                if isinstance(mot_tracker, TrackingManager):
                    invalid_ids.update(mot_tracker.invalid_ids)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("SORT frame=%d active=%d recovered=%s expired=%s invalid=%s",
                                     frame_nmr, len(mot_tracker.tracks), mot_tracker.recovered_ids,
                                     mot_tracker.expired_ids, sorted(invalid_ids))

                stage = "crossing evaluation"
                crossing_detector.expire(frame_nmr)
                for track_id in invalid_ids:
                    crossing_detector.mark_uncertain(track_id, 'invalid or ambiguous tracking output')
                visible_candidates = set()

                stage = "frame annotation"
                # Drawing bounding boxes
                for x1, y1, x2, y2, car_id in track_ids:
                    stage = "crossing evaluation"
                    observation = crossing_detector.observe_bbox(
                        car_id, (x1, y1, x2, y2), frame_nmr, frame_signal,
                        signal_version, timestamp_ms,
                    )
                    if observation['anchor'] is None:
                        logger.debug("Uncertain track at frame %d: %s", frame_nmr, observation['reason'])
                        continue
                    car_id = observation['track_id']
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    w, h = x2 - x1, y2 - y1
                    trigger_x, trigger_y = map(int, observation['anchor'])

                    # Candidates persist regardless of later signal/OCR changes.
                    if car_id in crossing_detector.events:
                        visible_candidates.add(car_id)

                    # red for violation
                    # green for normal
                    stage = "frame annotation"
                    color = (0, 0, 255) if car_id in visible_candidates else (0, 255, 0)
                    if color == (0, 0, 255):
                        cv2.circle(frame, (trigger_x, trigger_y), 2, color, -1)
                    bbox = [x1, y1, w, h]
                    cvzone.cornerRect(frame, bbox, l=8, t=2, rt=1, colorR=color)

                    # Show license plate text (if known)
                    if car_id in plate_memory:
                        plate_text = plate_memory[car_id]
                        cvzone.putTextRect(frame, plate_text, (x1, y1 - 25),
                                           scale=1, thickness=2, colorR=color, colorT=(255, 255, 255))

                # Enrich unresolved candidates, including after red becomes green.
                # Stop OCR for a track once its candidate has a recognized plate.
                pending_ocr = {car_id for car_id in visible_candidates
                               if crossing_detector.events[car_id]['license_plate'] is None}
                if pending_ocr:
                    stage = "license plate inference"
                    license_plates = license_plate_detector_model(inference_frame)[0]
                    for license_plate in license_plates.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = license_plate

                        stage = "license plate association"
                        # assign license plate to car
                        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                        if car_id != -1 and car_id in pending_ocr:

                            stage = "license plate preprocessing"
                            # crop license plate
                            license_plate_crop = inference_frame[int(y1):int(y2), int(x1): int(x2), :]
                            if license_plate_crop.size == 0:
                                raise ValueError("License plate crop is empty")

                            # process license plate
                            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)


                            # read license plate number
                            stage = "OCR processing"
                            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                            if crossing_detector.enrich_event(car_id, license_plate_text, license_plate_text_score):
                                plate_memory[car_id] = crossing_detector.events[car_id]['license_plate']
                                pending_ocr.discard(car_id)
                                # results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                #                               'license_plate': {'bbox': [x1, y1, x2, y2],
                                #                                                 'text': license_plate_text,
                                #                                                 'bbox_score': score,
                                #                                                 'text_score': license_plate_text_score}}

                stage = "JPEG encoding"
                if frame is None or frame.size == 0:
                    raise ValueError("Cannot encode an empty frame")
                try:
                    encoded, buffer = cv2.imencode('.jpg', frame)
                except cv2.error as error:
                    logger.warning("JPEG encoding raised at frame %d: %s", frame_nmr, error)
                    encoded, buffer = False, None
                if not encoded or buffer is None or buffer.size == 0:
                    encoding_failures += 1
                    logger.warning("JPEG encoding failed at frame %d (%d/%d)",
                                   frame_nmr, encoding_failures, MAX_ENCODING_FAILURES)
                    if encoding_failures >= MAX_ENCODING_FAILURES:
                        logger.error("Stopping video after repeated JPEG encoding failures")
                        shutdown_resources()
                        return
                    continue
                encoding_failures = 0
                jpeg = buffer.tobytes()
            except Exception:
                # Inference/tracking/annotation errors may leave shared state
                # inconsistent. Stop once instead of retrying indefinitely.
                logger.exception("Stopping video at frame %d during %s", frame_nmr, stage)
                shutdown_resources()
                return

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')


@app.route('/', methods=["GET"])
def index():
    return render_template("index.html", light=traffic_light)


@app.route('/video')
def video():
    try:
        initialize_resources()
    except Exception:
        logger.exception("Video initialization failed")
        return Response("Video unavailable. See server logs for details.",
                        status=503, mimetype="text/plain")
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_light/<color>', methods=['POST'])
def set_light(color):
    global traffic_light, _signal_version
    if color in ["red", "green"]:
        with _signal_lock:
            if traffic_light != color:
                traffic_light = color
                _signal_version += 1
    return redirect(url_for("index"))


@app.route("/violations")
def get_violations():
    global plate_memory
    # Convert dict to list of plate numbers
    violations_per_frame = list(plate_memory.values())
    return jsonify(violations_per_frame)


def main():
    logging.basicConfig(level=logging.INFO)
    try:
        initialize_resources()
        app.run(debug=False, use_reloader=False)
        return 0
    except Exception:
        logger.exception("Application startup or server failed")
        return 1
    finally:
        shutdown_resources()


if __name__ == "__main__":
    raise SystemExit(main())
