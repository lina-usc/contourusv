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

def clean_spec(plt_dat):

    # Apply median filter
    filtered_data = ndimage.median_filter(plt_dat, 3)

    # Normalize data for thresholding (0-255)
    norm_image = cv2.normalize(
        filtered_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu's Thresholding
    ret, thresh_img = cv2.threshold(
        norm_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Enhance Contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(thresh_img)

    # Use a kernel for morphological operations
    kernel = np.ones((10, 10), np.uint8)

    # Erosion followed by dilation (closing)
    closing_img = cv2.morphologyEx(enhanced_img, cv2.MORPH_CLOSE, kernel)

    return closing_img