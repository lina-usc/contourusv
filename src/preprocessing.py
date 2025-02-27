import cv2
import numpy as np
from scipy import ndimage
from sklearn.decomposition import NMF
import librosa

def clean_spec(Sxx):
    """
    Preprocess spectrogram data for USV detection.

    Processing steps:
    1. Median filtering
    2. Normalization (0-255)
    3. Otsu's thresholding
    4. CLAHE contrast enhancement
    5. Morphological closing

    Parameters
    ----------
    Sxx : ndarray
        Input spectrogram (2D array in dB scale)

    Returns
    -------
    ndarray
        Preprocessed binary image (2D uint8 array)
    """

    # Apply median filter
    filtered_data = ndimage.median_filter(Sxx, 3)

    # blurred_data = cv2.GaussianBlur(filtered_data, (5, 5), 0)

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
