from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


def process_experiment(experiment, tests):
    for test in tests:
        print(f"Processing {experiment} {test} experiment...")
        # files_path = Path("/Users/evana_anis/Library/CloudStorage/OneDrive-UniversityofSouthCarolina"
        #                   "/Devin/raw") / experiment / test
        files_path = Path(
            "/Users/evana_anis/Desktop/VSCode") / experiment / test
        audio_files = sorted(list(files_path.rglob(
            "*.wav")) + list(files_path.rglob("*.WAV")))
        for audio_file in audio_files:
            annotations = []  # Reset annotations for each audio file
            # Load the image
            image_file_pattern = f"data/{experiment}/{test}/spectrograms/cleaned/{audio_file.stem}/{audio_file.stem}_*.png"
            for image_file in sorted(Path(".").glob(image_file_pattern)):
                start_time, end_time = map(
                    float, image_file.stem.split('_')[-2:])
                image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
                # Apply a threshold to detect USVs
                _, thresholded_image = cv2.threshold(
                    image, 50, 255, cv2.THRESH_BINARY)

                # Find contours in the thresholded image
                contours, _ = cv2.findContours(
                    thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Initialize a list to hold the details of each detected USV
                usv_details = []

                # Process each contour
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    duration_start = start_time + \
                        (x / image.shape[1]) * (end_time - start_time)
                    duration_end = start_time + \
                        ((x + w) / image.shape[1]) * (end_time - start_time)
                    freq_start = 110000 - \
                        ((y + h) / image.shape[0]) * (110000 - 20000)
                    freq_end = 110000 - \
                        (y / image.shape[0]) * (110000 - 20000)

                    duration = duration_end - duration_start
                    usv_details.append({
                        'bounding_box': (x, y, w, h),
                        'duration': (x, x + w),
                        'duration_start': duration_start,
                        'duration_end': duration_end,
                        'freq_start': freq_start,
                        'freq_end': freq_end
                    })

                    annotations.append({
                        'file': audio_file.name,
                        'begin_time': duration_start,
                        'end_time': duration_end,
                        'low_freq': freq_start,
                        'high_freq': freq_end,
                        'duration': duration,
                        'USV_TYPE': '22khz'
                    })
                # Filter USVs for 50 kHz calls
                usv_details = [usv for usv in usv_details if usv['freq_start']
                               >= 20000 and usv['freq_end'] <= 80000]
                # Filter USVs with low durations
                usv_details = [
                    usv for usv in usv_details if usv['duration'][1] - usv['duration'][0] > 20]
                # Filter USVs with high durations
                usv_details = [
                    usv for usv in usv_details if usv['duration'][1] - usv['duration'][0] < 120]
                # Annotate the image with the updated labeling function
                image_with_annotations = cv2.cvtColor(
                    image, cv2.COLOR_GRAY2BGR)
                for usv in usv_details:
                    x, y, w, h = usv['bounding_box']
                    cv2.rectangle(image_with_annotations, (x, y),
                                  (x + w, y + h), (0, 255, 0), 2)

                final_image = cv2.cvtColor(
                    image_with_annotations, cv2.COLOR_BGR2RGB)
                output_dir = Path(
                    f'data/{experiment}/{test}/spectrograms/detected/{audio_file.stem}')
                # Ensure output directory exists
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                fig, ax = plt.subplots(figsize=(20, 10))
                ax.imshow(final_image, aspect='auto', extent=[
                    start_time, end_time, 20000, 110000])
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
                plt.savefig(
                    f"{output_dir}/{Path(image_file).stem}_labeled.png", bbox_inches='tight', pad_inches=0)
                plt.close()

            # Save annotations to a DataFrame for the current audio file
            df = pd.DataFrame(annotations)
            df = df.sort_values('begin_time')
            # Filter out USVs with duration less than 0.002 seconds and greater than 0.33 seconds
            df = df[(df['duration'] >= 0.002) & (df['duration'] <= 0.33)]
            # Filter out annotations with frequencies less than 20 kHz and more than 80 kHz
            df = df[(df['low_freq'] >= 20000) & (df['high_freq'] <= 80000)]
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


def main():
    # Map each experiment to its corresponding tests
    experiment_tests_mapping = {
        # 'EPHYS': ['ACQ' , 'CUE'],
        # 'PTSD16': ['ACQ']
        # 'PTSD18': ['GEN'],
        # 'PTSD20': ['ACQ']
        # 'USVSEG': ['rat_distressed']
        # 'USVSEG': ['rat_pleasant']
        'usvseg_data': ['gerbil']
    }
    for experiment, tests in experiment_tests_mapping.items():
        # Measure the time taken to process the experiment
        start_time = time.time()
        process_experiment(experiment, tests)
        end_time = time.time()
        print(
            f"Finished processing {experiment} experiment in {end_time - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
