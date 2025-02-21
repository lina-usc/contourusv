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

from spectrogram import gen_spec
from preprocessing import clean_spec
from detection import detect_contours

def gen_figures(file_name, experiment, test, overlap=3, winlen=10, freq_min=15, freq_max=115, wsize=2500, th_perc=99.5):
    stft_df, v_max, sfreq, plt_dat = gen_spec(file_name, freq_min, freq_max, wsize, th_perc)
    nb_samples = None
    for start_time in np.arange(0, stft_df.columns[-1], winlen):
        end_time = start_time+winlen+overlap
        plt_dat = stft_df.loc[(stft_df.index > freq_min) & (
            stft_df.index < freq_max), (stft_df.columns > start_time) & (stft_df.columns < end_time)]
        if nb_samples is None:
            nb_samples = plt_dat.shape[1]
        elif nb_samples != plt_dat.shape[1]:
            tmp_dat = np.zeros([plt_dat.shape[0], nb_samples])
            tmp_dat[:, :plt_dat.shape[1]] = plt_dat.values
            plt_dat = tmp_dat

        cleaned_image = clean_spec(plt_dat)
        final_image, annotations = detect_contours(cleaned_image, start_time, end_time, file_name)


        output_dir = Path(
            f'data/{experiment}/{test}/spectrograms/detected_test/{file_name.stem}')
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(final_image, aspect='auto', extent=[
            start_time, end_time, 0, 120000])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        plt.savefig(
            f"{output_dir}/{Path(file_name).with_suffix('').name}_{start_time}_{end_time}.png",
            bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    # Save annotations to a DataFrame for the current audio file
    df = pd.DataFrame(annotations)
    df = df.sort_values('begin_time')
    # Convert the numerical columns to 0.2f precision
    df = df.round(2)
    # Filter duplicate annotations based on the begin_time and end_time columns
    df = df.drop_duplicates(
        subset=['begin_time', 'end_time'], keep='first')

    output_annotation_file = f'data/{experiment}/{test}/contour_detections/{audio_file.stem}.csv'
    Path(output_annotation_file).parent.mkdir(
        parents=True, exist_ok=True)
    df.to_csv(output_annotation_file, sep='\t', index=False)
    print(f'Saved annotations to {output_annotation_file}')

if __name__ == "__main__":

    # Map each experiment to its corresponding tests
    experiment_tests_mapping = {
        'PTSD16': ['ACQ']
    }
    root_path = Path("/Users/evana_anis/Library/CloudStorage/OneDrive-UniversityofSouthCarolina"
                    "/Devin/raw")

    for experiment, tests in experiment_tests_mapping.items():
        for test in tests:
            files_path = Path(root_path) / experiment / test
            audio_files = sorted(list(files_path.rglob(
                "*.wav")) + list(files_path.rglob("*.WAV")))
            # Measure the time taken to process the experiment
            start_time = time.time()
            for audio_file in audio_files:
                gen_figures(audio_file, experiment, test)
            end_time = time.time()
            print(
                f"Generated spectrograms in {end_time-start_time} seconds.")
