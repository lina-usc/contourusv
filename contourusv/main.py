import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
from scipy.signal import spectrogram, butter, filtfilt
from sklearn.decomposition import NMF, FastICA
from preprocessing import clean_spec_imp, clean_spec_orig
from evaluation import run_evaluation
from detection import detect_contours
from generate_annotation import generate_annotations
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter("ignore", ConvergenceWarning)

from sklearn.decomposition import FastICA

def use_ICA(Sxx):
    """
    Perform Independent Component Analysis (ICA) on the input spectrogram.

    Parameters:
    -----------
    Sxx : ndarray
        Input spectrogram to process.

    Returns:
    --------
    ndarray
        Transformed spectrogram after ICA.
    """

    Sxx = np.nan_to_num(Sxx, nan=0.0, posinf=255, neginf=0).astype(np.float32)

    ica = FastICA(n_components=None, random_state=0)
    transformed_Sxx = ica.fit_transform(Sxx.T)  # Apply ICA transformation

    return transformed_Sxx

def use_NMF_Small(Sxx, num_splits=120, n_components=25):
    """
    Perform Non-negative Matrix Factorization (NMF) on the input spectrogram 
    by splitting it into parts and applying NMF with specified components on each block.
    This segmentation allows better capture of properties of the signal when it is non-stationary.

    Parameters:
    -----------
    Sxx : ndarray
        Input spectrogram to process.
    num_splits : int, optional
        Number of segments to divide the spectrogram into (default: 80).
    n_components : int, optional
        Number of NMF components (default: 25).

    Returns:
    --------
    ndarray
        Transformed spectrogram after NMF.
    """

    # Shift the spectrogram to make all values non-negative
    min_value = np.min(Sxx)
    if min_value < 0:
        Sxx = Sxx - min_value  # Shift all values up to be non-negative


    # Determine split size
    split_size = max(n_components, Sxx.shape[1] // num_splits)  # Ensure at least `n_components` columns

    transformed_parts = []

    for i in range(0, Sxx.shape[1], split_size):
        end_idx = min(i + split_size, Sxx.shape[1])  # Avoid out-of-bounds
        
        Sxx_part = Sxx[:, i:end_idx]  # Extract segment
        
        # Pad small segments with zeros or mean value to ensure correct dimensions
        if Sxx_part.shape[1] < n_components:
            print(f"Padding segment {i}-{end_idx} to meet {n_components} components")
            # Padding with zeros
            padding = np.zeros((Sxx_part.shape[0], n_components - Sxx_part.shape[1]))
            Sxx_part = np.hstack((Sxx_part, padding))  # Add padding to the segment

        # Use `nndsvd` if possible, otherwise fall back to `random`
        init_method = 'nndsvd' if Sxx_part.shape[1] >= n_components else 'random'
        if init_method == 'nndsvd':
            max_iter = 100
        else:
            max_iter = 500
        
        # Apply NMF
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = NMF(n_components=n_components, init=init_method, random_state=0, max_iter=max_iter)
        W = model.fit_transform(Sxx_part)
        H = model.components_

        # Reconstruct the matrix segment
        reconstructed_Sxx_part = np.dot(W, H)

        transformed_parts.append(reconstructed_Sxx_part)

    # Concatenate along the time axis (columns)
    reconstructed_Sxx = np.hstack(transformed_parts)

    return reconstructed_Sxx


def use_NMF(Sxx, n_components=30):
    """
    Perform Non-negative Matrix Factorization (NMF) on the input spectrogram.

    Parameters:
    -----------
    Sxx : ndarray
        Input spectrogram to process.
    n_components : int
        Components to split for NMF (default: 30)

    Returns:
    --------
    ndarray
        Transformed spectrogram after NMF.
    """

    # Shift the spectrogram to make all values non-negative
    min_value = np.min(Sxx)
    if min_value < 0:
        Sxx = Sxx - min_value  # Shift all values up to be non-negative

    # Apply NMF 
    # Notes: 
    # Performs well on Gerbil, MP
    # Minimal Change on C57
    # Performs poor on Mouse_B6PUP
    # Seems to take a long time to runimport warnings
    # n = 30 determined by trial and error, 30 provided consistent highest results

    model = NMF(n_components=n_components, init='nndsvd', random_state=0, max_iter=100)
    W = model.fit_transform(Sxx)  # W is the transformed data (lower-dimensional)
    H = model.components_         # H is the components (basis)

    # Reconstruct the full matrix
    reconstructed_Sxx = np.dot(W, H)  # Reconstructed spectrogram (W * H)

    return reconstructed_Sxx



def low_pass_filter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth low-pass filter to input data.

    Parameters
    ----------
    data : ndarray
        Input signal to be filtered
    cutoff : float
        Cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int, optional
        Filter order (default: 5)

    Returns
    -------
    ndarray
        Filtered output signal
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def run_detection(root_path, file_name, experiment, trial, overlap=3,
                  winlen=10, freq_min=15, freq_max=115, wsize=2500, th_perc=95, processing='none', overlapsize=.25):
    """
    Process audio file to detect ultrasonic vocalizations (USVs).

    Parameters
    ----------
    root_path : Path
        Root directory for input/output data
    file_name : Path
        Path to audio file to process
    experiment : str
        Name of the experiment
    trial : str
        Name of the trial/condition
    overlap : int, optional
        Overlap between processing windows in seconds (default: 3)
    winlen : int, optional
        Window length for processing in seconds (default: 10)
    freq_min : int, optional
        Minimum frequency for USV detection in kHz (default: 15)
    freq_max : int, optional
        Maximum frequency for USV detection in kHz (default: 115)
    wsize : int, optional
        Spectrogram window size (default: 2500)
    th_perc : float, optional
        Percentile threshold for noise reduction (default: 95)
    processing : str, optional
        Type of preprocessing to apply Otsu/Adaptive(default: 'adaptive')
    overlapsize: int, optional
        Defines size of overlap window as a percentage  (defailt: .25)

    Returns
    -------
    None
        Outputs:
        - Annotated spectrogram images in {root_path}/output/spectrograms/
        - Detection annotations in {root_path}/output/contour_detections/
    """
    sfreq, data = wavfile.read(file_name)
    print(f"Processing {file_name.stem}... Sampling Frequency: {sfreq} Hz...")

    # # Apply low-pass filter
    data = low_pass_filter(data, cutoff=120000, fs=sfreq)

    # Normalize the audio data
    data = data / np.max(np.abs(data))

    # Calculate the total time in seconds
    total_time = len(data) / sfreq

    # Generate the time array
    time = np.arange(len(data)) / sfreq

    annotations = []  # Reset annotations for each audio file

    # Loop through the entire audio in 10-second windows with a 3-second overlap
    for start_time in np.arange(0, total_time, winlen):
        end_time = start_time + winlen + overlap
        if end_time > total_time:
            end_time = total_time

        # Extract the segment of audio for this time window
        start_sample = int(start_time * sfreq)
        end_sample = int(end_time * sfreq)
        data_segment = data[start_sample:end_sample]
        
        #Define spectrogram parameters
        window = 'hann'
        nperseg = 512
        noverlap = int(nperseg * overlapsize)
        nfft = 512
        scaling = 'density'
        mode = 'magnitude'

        # Compute spectrogram
        f, t, Sxx = spectrogram(data_segment, fs=sfreq, window=window, nperseg=nperseg,
                                noverlap=noverlap, nfft=nfft, scaling=scaling, mode=mode)
        
        Sxx = 10 * np.log10(Sxx + 1e-10)

        # Convert frequencies to kHz
        f = f / 1000

        # Noise reduction: apply a decibel threshold
        noise_floor = np.percentile(Sxx, th_perc)
        Sxx[Sxx < noise_floor] = noise_floor

        if (processing != "Otsu"):
            Sxx = use_NMF_Small(Sxx)

        if(processing == "Otsu"):
            cleaned_image = clean_spec_orig(Sxx)
        else:
            cleaned_image = clean_spec_imp(Sxx)

        final_image, annotations = detect_contours(cleaned_image, start_time, end_time, freq_min, freq_max, file_name, annotations, processing=processing)

        output_dir = Path(
            f'{root_path}/output/{experiment}/{trial}/spectrograms/{file_name.stem}')
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Visualize the spectrograms
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(final_image, aspect='auto', extent=[
            start_time, end_time, 0, 120], origin='lower')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (kHz)')
        plt.savefig(
            f"{output_dir}/{Path(file_name).with_suffix('').name}_{start_time}_{end_time}.png",
            bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    # Save annotations to a DataFrame for the current audio file
    df = pd.DataFrame(annotations)

    # Proceed only if detections were made
    if not df.empty:
        df = df.sort_values('begin_time')
        df = df.round(2)
        df = df.drop_duplicates(subset=['begin_time', 'end_time'], keep='first')

        output_annotation_file = f'{root_path}/output/{experiment}/{trial}/contour_detections/{audio_file.stem}.csv'
        Path(output_annotation_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_annotation_file, sep='\t', index=False)
        print(f'Saved annotations to {output_annotation_file}')
    else:
        print(f'No USVs detected in {audio_file.name}, skipping CSV export.')


if __name__ == "__main__":
    """
    USV Detection Pipeline main entry point.

    Processes audio files to detect ultrasonic vocalizations, generates ground truth
    annotations, runs evaluation metrics, and tracks computational resources.

    Command Line Arguments:
    --root_path : Root directory containing experiment data
    --experiment : Name of the experiment (e.g., PTSD16)
    --trial : Trial/condition name (e.g., ACQ)
    --file_ext : Annotation file extension (.html, .xlsx, .csv)
    [Other arguments...]
    """
    parser = argparse.ArgumentParser(description="USV Detection Pipeline")
    parser.add_argument("--root_path", type=str, default="/Users/username/data",
                        help="Root path to the experiment data", required=True)
    parser.add_argument("--experiment", type=str, default="PTSD16", help="Experiment name")
    parser.add_argument("--trial", type=str, default="ACQ", help="trial name")
    parser.add_argument("--overlap", type=int, default=3, help="Overlap duration")
    parser.add_argument("--winlen", type=int, default=10, help="Window length")
    parser.add_argument("--freq_min", type=int, default=15, help="Minimum frequency")
    parser.add_argument("--freq_max", type=int, default=115, help="Maximum frequency")
    parser.add_argument("--wsize", type=int, default=2500, help="Window size")
    parser.add_argument("--th_perc", type=float, default=95, help="Threshold percentage")
    parser.add_argument("--file_ext", type=str, default='.html', required= True, help="File extension to process (.html, .xlsx, .csv)")
    parser.add_argument("--processing", type=str, default='adaptive', help="Processing method (Otsu/Adaptive)")

    args = parser.parse_args()

    root_path = Path(args.root_path)
    experiment = args.experiment
    trial = args.trial
    file_ext = args.file_ext

    ac_kwargs = {
        "overlap": args.overlap,
        "winlen": args.winlen,
        "freq_min": args.freq_min,
        "freq_max": args.freq_max,
        "wsize": args.wsize,
        "th_perc": args.th_perc,
        "processing": args.processing
    }

    files_path = Path(root_path) / experiment / trial
    output_path = Path(
        f'{root_path}/output/{experiment}/{trial}')
    Path(output_path).mkdir(parents=True, exist_ok=True)

    audio_files = sorted(list(files_path.rglob(
        "*.wav")) + list(files_path.rglob("*.WAV")))

    # Measure the time and energy taken to execute the pipeline
    start_time = time.time()
    tracker = EmissionsTracker(output_dir=output_path)
    tracker.start()

    # Run detection for each audio file
    for audio_file in tqdm(audio_files, desc=f"Running Detection on audio files for {experiment} {trial}"):
        run_detection(root_path, audio_file, experiment, trial, **ac_kwargs)
    
    # Generate ground truth annotations
    generate_annotations(experiment, trial, root_path, file_ext)

    # Run evaluation
    run_evaluation(experiment, trial, root_path)

    end_time = time.time()
    total_time = end_time - start_time

    emissions: float = tracker.stop()
    total_energy = tracker.final_emissions_data.energy_consumed

    print(f"ContourUSV Detection Pipeline Executed for {experiment} {trial}")
    print("---------------------------------------------------")
    print(f"ContourUSV_Execution_Time_(s) = {total_time:.3f}")
    print(f"ContourUSV_Carbon_Emissions_(kgCO2) = {emissions:.3f}")
    print(f"ContourUSV_Total_Energy_Consumed_(kWh) = {total_energy:.3f}")