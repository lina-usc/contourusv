import cv2
import pandas as pd
import os
# import remove_suffix  # Removed because not needed


def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_length=5):
    x1, y1 = pt1
    x2, y2 = pt2

    # horizontal top
    for x in range(x1, x2, dash_length*2):
        cv2.line(img, (x, y1), (min(x+dash_length, x2), y1), color, thickness)
    # horizontal bottom
    for x in range(x1, x2, dash_length*2):
        cv2.line(img, (x, y2), (min(x+dash_length, x2), y2), color, thickness)
    # vertical left
    for y in range(y1, y2, dash_length*2):
        cv2.line(img, (x1, y), (x1, min(y+dash_length, y2)), color, thickness)
    # vertical right
    for y in range(y1, y2, dash_length*2):
        cv2.line(img, (x2, y), (x2, min(y+dash_length, y2)), color, thickness)

def detect_contours(cleaned_image, start_time, end_time, freq_min, freq_max, 
                    file_name, annotations, call_type_defs=None, processing="adaptive",):
    """
    Detect and classify USVs in cleaned spectrogram images, with optional
    overlay of manual annotations for evaluation.

    Parameters
    ----------
    cleaned_image : ndarray
        Preprocessed spectrogram image (2D array)
    start_time : float
        Start time of current processing window (seconds)
    end_time : float
        End time of current processing window (seconds)
    freq_min : float
        Minimum frequency bound for detection (kHz)
    freq_max : float
        Maximum frequency bound for detection (kHz)
    file_name : Path
        Source audio file path
    annotations : list
        List to accumulate detection annotations
    call_type_defs : dict
        Call type definitions
    processing : str
        Thresholding method ("adaptive" or "Otsu")

    Returns
    -------
    tuple
        (annotated_image, updated_annotations)
        annotated_image: RGB image with detection bounding boxes
        updated_annotations: List of annotation dictionaries
    """
    if call_type_defs is None:
        call_type_defs = {
            "22kHz": {"freq_min": 15, # 25kh works to eliminate false low call detections, but is incorrect logically
                      "freq_max": 45,
                      "freq_span_max": 10,
                      "duration_min": 0.03,
                      "duration_max": 3.0},
            "50kHz": {"freq_min": 40,
                      "freq_max": 80,
                      "freq_span_max": 40,
                      "duration_min": 0.01,
                      "duration_max": 0.3},
        }

    # Re-apply Otsu's Thresholding if using Otsus
    if processing == "Otsu":
        ret, thresholded_image = cv2.threshold(
            cleaned_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thresholded_image = cleaned_image

    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    usv_details = []

    # Process each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        duration_start = start_time + (x / thresholded_image.shape[1]) * (end_time - start_time)
        duration_end = start_time + ((x + w) / thresholded_image.shape[1]) * (end_time - start_time)
        freq_start = freq_min + (y / thresholded_image.shape[0]) * (freq_max - freq_min)
        freq_end = freq_min + ((y + h) / thresholded_image.shape[0]) * (freq_max - freq_min)

        duration = duration_end - duration_start
        freq_span = freq_end - freq_start
        
        for call_type, call_def in call_type_defs.items():
            if (freq_start > call_def["freq_min"] and 
                freq_end < call_def["freq_max"] and 
                freq_span < call_def["freq_span_max"] and
                call_def["duration_max"] >= duration >= call_def["duration_min"]):

                usv_details.append({
                    'bounding_box': (x, y, w, h),
                    'duration': (x, x + w),
                    'duration_start': duration_start,
                    'duration_end': duration_end,
                    'freq_start': freq_start,
                    'freq_end': freq_end
                })

                annotations.append({
                    'file': file_name.name,
                    'begin_time': duration_start,
                    'end_time': duration_end,
                    'low_freq': freq_start,
                    'high_freq': freq_end,
                    'duration': duration,
                    'USV_TYPE': call_type
                })

    # Annotate detections
    image_with_annotations = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
    for usv in usv_details:
        x, y, w, h = usv['bounding_box']
        cv2.rectangle(image_with_annotations, (x, y),
                      (x + w, y + h), (0, 255, 0), 1)  # green boxes
        
    manual_csv = str(file_name).replace(".wav", ".csv")

    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    # Overlay manual annotations
    if os.path.exists(manual_csv):
        manual_annots = pd.read_csv(manual_csv, header=None, names=["begin_time", "end_time"])
        manual_subset = manual_annots[
            (manual_annots["end_time"] >= start_time) & 
            (manual_annots["begin_time"] <= end_time)
        ].reset_index(drop=True)

        for idx, row in manual_subset.iterrows():
            # map times to x coordinates for markers
            x1 = int(((row["begin_time"] - start_time) / (end_time - start_time)) * thresholded_image.shape[1])
            x2 = int(((row["end_time"] - start_time) / (end_time - start_time)) * thresholded_image.shape[1])
            y1, y2 = 0, thresholded_image.shape[0]  # full height

            color = colors[idx % len(colors)]  # cycle through colors
            draw_dashed_rect(image_with_annotations, (x1, y1), (x2, y2), color, thickness=2, dash_length=5)

    final_image = cv2.cvtColor(image_with_annotations, cv2.COLOR_BGR2RGB)
    
    return final_image, annotations
