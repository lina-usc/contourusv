import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import pandas as pd
from pathlib import Path
import time
import cv2
from scipy import ndimage
from PIL import Image as im
import seaborn as sns

def detect_contours(cleaned_image, start_time, end_time, freq_min, freq_max, file_name, annotations):

    # Re-apply Otsu's Thresholding
    ret, thresholded_image = cv2.threshold(
        cleaned_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to hold the details of each detected USV
    usv_details = []

    # Process each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        duration_start = start_time + \
            (x / thresholded_image.shape[1]) * (end_time - start_time)
        duration_end = start_time + \
            ((x + w) /
                thresholded_image.shape[1]) * (end_time - start_time)
        freq_start = freq_min + \
            (y / thresholded_image.shape[0]) * (freq_max - freq_min)
        freq_end = freq_min + \
            ((y + h) / thresholded_image.shape[0]) * (freq_max - freq_min)

        duration = duration_end - duration_start
        # Frequency span
        freq_span = freq_end - freq_start
        
        # For 22 kHz USVs
        if freq_start > 15 and freq_end < 45 and freq_span < 10:
            if duration > 0.03 and duration < 3.0:
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
                    'USV_TYPE': '22khz'
                })

        # For 50 kHz USVs
        if freq_start > 40 and freq_end < 80 and freq_span < 10:
            if duration > 0.01 and duration < 0.3:
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
                    'USV_TYPE': '50khz'
                })
    # Annotate the image with the updated labeling function
    image_with_annotations = cv2.cvtColor(
        thresholded_image, cv2.COLOR_GRAY2BGR)
    for usv in usv_details:
        x, y, w, h = usv['bounding_box']
        cv2.rectangle(image_with_annotations, (x, y),
                        (x + w, y + h), (0, 255, 0), 1)

    final_image = cv2.cvtColor(
        image_with_annotations, cv2.COLOR_BGR2RGB)
    
    return final_image, annotations