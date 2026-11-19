"""Plate helper compatibility API; heavyweight optional imports stay local."""
import logging
from plate_recognition import (OCRConfig, associate_plate, correction_candidates,
                               rank_ocr, validate_plate)

logger = logging.getLogger(__name__)
# Cached assets only. Missing/corrupt EasyOCR weights fail without downloading.
reader = None


def get_reader():
    global reader
    if reader is None:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, download_enabled=False)
    return reader


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text, config=None):
    return len(correction_candidates(text, 1.0, config)) == 1


def format_license(text, config=None):
    candidates = correction_candidates(text, 1.0, config)
    return candidates[0].normalized_text if len(candidates) == 1 else None


def read_license_plate(license_plate_crop, config=None):
    """Structured result, also unpackable as the historical (text, score) pair."""
    config = config or OCRConfig()
    detections = get_reader().readtext(license_plate_crop)
    if config.diagnostic:
        # Explicit local opt-in only; do not enable for routine processing.
        logger.debug('Local OCR diagnostic outputs: %r', detections)
    return rank_ocr(detections, config)


def get_car(license_plate, vehicle_track_ids):
    return associate_plate(license_plate, vehicle_track_ids)


def load_model(model_path, key):
    """
    Load the model from the given path.
    Args:
        model_path:
        key:

    Returns:

    """
    import streamlit as st
    from ultralytics import YOLO

    if st.session_state[key] is None:
        try:
            with st.spinner(f"Loading {key}..."):
                st.session_state[key] = YOLO(model_path)
            st.success(f"{key} loaded successfully")
        except Exception as e:
            st.error(f"Failed to load {key} : {e}")
            st.stop()
