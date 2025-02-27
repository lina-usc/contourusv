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
from sklearn.decomposition import NMF

from preprocessing import clean_spec
from evaluation import run_evaluation
from detection import detect_contours
from generate_annotation import generate_annotations

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
                  winlen=10, freq_min=15, freq_max=115, wsize=2500, th_perc=95):
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

    data = data.reshape(-1, 1)  # Reshape to (n_samples, 1)

    data = np.maximum(data, 0)  # Remove negative values

    model = NMF(n_components=2, init='random', random_state=0)
    data = model.fit_transform(data)  # Basis matrix
    H = model.components_  # Activation matrix

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
        noverlap = 1
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

        cleaned_image = clean_spec(Sxx)
        final_image, annotations = detect_contours(cleaned_image, start_time, end_time, freq_min, freq_max, file_name, annotations)

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

    # # Save annotations to a DataFrame for the current audio file
    # df = pd.DataFrame(annotations)
    # df = df.sort_values('begin_time')
    # # Convert the numerical columns to 0.2f precision
    # df = df.round(2)
    # # Filter duplicate annotations based on the begin_time and end_time columns
    # df = df.drop_duplicates(
    #     subset=['begin_time', 'end_time'], keep='first')

    # output_annotation_file = f'{root_path}/output/{experiment}/{trial}/contour_detections/{audio_file.stem}.csv'
    # Path(output_annotation_file).parent.mkdir(
    #     parents=True, exist_ok=True)
    # df.to_csv(output_annotation_file, sep='\t', index=False)
    # print(f'Saved annotations to {output_annotation_file}')

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
        "th_perc": args.th_perc
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
    
    # # Generate ground truth annotations
    # generate_annotations(experiment, trial, root_path, file_ext)

    # Run evaluation
    # run_evaluation(experiment, trial, root_path)

    end_time = time.time()
    total_time = end_time - start_time

    emissions: float = tracker.stop()
    total_energy = tracker.final_emissions_data.energy_consumed

    print(f"ContourUSV Detection Pipeline Executed for {experiment} {trial}")
    print("---------------------------------------------------")
    print(f"ContourUSV_Execution_Time_(s) = {total_time:.3f}")
    print(f"ContourUSV_Carbon_Emissions_(kgCO2) = {emissions:.3f}")
    print(f"ContourUSV_Total_Energy_Consumed_(kWh) = {total_energy:.3f}")