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


def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def gen_figures(file_name, experiment, test, output_dir, overlap=3, winlen=10, freq_min=20, freq_max=110, wsize=2500, th_perc=99.5):
    sfreq, data = wavfile.read(file_name)
    # # Apply low-pass filter
    # data = low_pass_filter(data, cutoff=120000, fs=sfreq)
    time = np.arange(len(data))/sfreq
    dt = wsize/sfreq/2

    tf = np.abs(mne.time_frequency.stft(data, wsize=wsize).squeeze())
    freqs = mne.time_frequency.stftfreq(wsize=wsize, sfreq=sfreq)

    stft_df = pd.DataFrame(data=tf[::-1], index=pd.Series(np.round(freqs[::-1]/1000, 1),
                           name="Frequencies (kHz)"), columns=np.arange(0, dt*tf.shape[1], dt))

    plt_dat = stft_df.loc[(stft_df.index > freq_min) &
                          (stft_df.index < freq_max), :]
    v_max = np.percentile(plt_dat, th_perc)

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

        # Apply median filter
        filtered_data = ndimage.median_filter(plt_dat, 3)

        # Normalize data for thresholding (0-255)
        norm_image = cv2.normalize(
            filtered_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Global Thresholding
        ret, global_thresh_img = cv2.threshold(
            norm_image, 5, 255, cv2.THRESH_BINARY)

        # Enhance Contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(global_thresh_img)

        # Use a kernel for morphological operations
        kernel = np.ones((10, 10), np.uint8)

        # Erosion followed by dilation (closing)
        closing_img = cv2.morphologyEx(enhanced_img, cv2.MORPH_CLOSE, kernel)

        # Ensure the directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        # plt.imsave(
        #     fname=f"{output_dir}/{Path(file_name).with_suffix('').name}_{start_time}_{end_time}.png", arr=closing_img, cmap="viridis", dpi=600)
        # Increase figure size for better visualization
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(closing_img, aspect='auto', cmap='viridis', extent=[
            start_time, end_time, freq_min, freq_max])
        plt.axis('off')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        # plt.colorbar(ax.imshow(closing_img, aspect='auto', cmap='viridis', extent=[
        #     start_time, end_time, freq_min, freq_max]), ax=ax)
        fig.savefig(f"{output_dir}/{Path(file_name).with_suffix('').name}_{start_time}_{end_time}.png",
                    bbox_inches='tight', pad_inches=0, dpi=600)
        plt.close(fig)


# Map each experiment to its corresponding tests
experiment_tests_mapping = {
    # 'EPHYS': ['ACQ', 'CUE']
    # 'PTSD16': ['ACQ']
    # 'PTSD18': ['GEN'],
    # 'PTSD20': ['ACQ']
    # 'VocalMat': ['wav_files']
    # 'Dyad': ['VOC591']
    # 'USVSEG': ['rat_pleasant']
    'usvseg_data': ['gerbil']
}
# root_path = Path("/Users/evana_anis/Library/CloudStorage/OneDrive-UniversityofSouthCarolina"
#                  "/Devin/raw")
# root_path2 = Path("/Users/evana_anis/Desktop/VSCode/dyad_usv_data/VOC591/")
root_path = Path(
    "/Users/evana_anis/Desktop/VSCode")
for experiment, tests in experiment_tests_mapping.items():
    for test in tests:
        files_path = Path(root_path) / experiment / test
        audio_files = sorted(list(files_path.rglob(
            "*.wav")) + list(files_path.rglob("*.WAV")))
        # Measure the time taken to process the experiment
        start_time = time.time()
        for audio_file in audio_files:
            output_dir = Path(
                f'data/{experiment}/{test}/spectrograms/cleaned/{audio_file.stem}')
            # Check if the detected annotation file already exists
            if (output_dir).exists():
                print(
                    f"Spectrograms for {audio_file.name} already exist. Skipping.")
                continue
            gen_figures(audio_file, experiment, test, output_dir)
        end_time = time.time()
        print(
            f"Generated spectrograms in {end_time-start_time} seconds.")
