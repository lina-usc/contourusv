import cv2
import numpy as np
from scipy import ndimage
from sklearn.decomposition import NMF
import librosa

def clean_spec_imp(Sxx):
    """
    Improved preprocessing of spectrogram data for USV detection, using adaptive theshholding. 

    Parameters
    ----------
    Sxx : ndarray
        Input spectrogram (2D array in dB scale)

    Returns
    -------
    ndarray
        Preprocessed binary image (2D uint8 array)
    """

    # Apply mild median filter to reduce noise while keeping signals
    filtered_data = ndimage.median_filter(Sxx, size=3)

    # Normalize to range (0-255)
    norm_image = cv2.normalize(filtered_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # CLAHE contrast enhancement with slightly reduced contrast
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(norm_image)

    # Adaptive Thresholding (Fine-tuned parameters)
    adaptive_img = cv2.adaptiveThreshold(
        enhanced_img, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        15, 15
    )

    # best results so far, 15, 15
    # 15, 19 seems to do well

    # Morphological processing to connect weakly detected signals
    # kernel = np.ones((1, 1), np.uint8)  # Moderate kernel size

    # final_img = cv2.morphologyEx(adaptive_img, cv2.MORPH_OPEN, kernel)

    return adaptive_img




def clean_spec_orig(Sxx):
    """
    Preprocess spectrogram data for USV detection using Otsu's threshholding and CLAHE contrast enhancement.

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