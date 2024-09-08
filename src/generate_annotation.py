import pandas as pd
from pathlib import Path
import time


def load_csv_usv(file_name):
    # Load the CSV file without headers, using only the first two columns
    data = pd.read_csv(file_name, header=None, names=[
                       'begin_time', 'end_time'], usecols=[0, 1])

    # Convert columns to float
    data['begin_time'] = data['begin_time'].astype(float)
    data['end_time'] = data['end_time'].astype(float)

    return data


def save_annotations(csv_files, audio_file_name, output_path, freq_min=15000, freq_max=80000):
    annotations = []
    for csv_file in csv_files:
        usv_data = load_csv_usv(csv_file)
        for index, row in usv_data.iterrows():
            annotations.append({
                'begin_file': audio_file_name,
                'begin_time': row['begin_time'],
                'end_time': row['end_time'],
                'low_freq': freq_min,
                'high_freq': freq_max
            })

    annotations_df = pd.DataFrame(annotations)
    output_file_name = output_path / (Path(audio_file_name).stem + '.csv')
    annotations_df.to_csv(output_file_name, sep='\t', index=False)


def generate_experiment_annotations(experiment, tests, root_path):
    for test in tests:
        files_path = Path(root_path) / experiment / test
        output_annotations_path = Path(
            f'data/{experiment}/{test}/ground_truth_annotations')
        output_annotations_path.mkdir(parents=True, exist_ok=True)
        print(f"Processing {experiment} {test} experiment...")
        csv_files = sorted(list(files_path.rglob("*.csv")))
        audio_files = sorted(list(files_path.rglob(
            "*.wav")) + list(files_path.rglob("*.WAV")))
        for audio_file in audio_files:
            matched_csv_files = [f for f in csv_files if f.stem.split(
                '_')[0:3] == audio_file.stem.split('_')[0:3]]
            print(f"Matched CSV files: {matched_csv_files}")
            save_annotations(matched_csv_files, audio_file,
                             output_annotations_path)


def main():
    # Map each experiment to its corresponding tests
    experiment_tests_mapping = {
        # 'USVSEG': ['rat_distressed'],
        # 'USVSEG': ['rat_pleasant']
        'usvseg_all': ['all']
    }
    root_path = Path("/Users/evana_anis/Desktop/VSCode")
    for experiment, tests in experiment_tests_mapping.items():
        # Measure the time taken to process the experiment
        start_time = time.time()
        generate_experiment_annotations(experiment, tests, root_path)
        end_time = time.time()
        print(
            f"Finished processing {experiment} experiment in {end_time - start_time:.2f} seconds.")
    print("Annotation files generation completed for all experiments and tests.")


if __name__ == '__main__':
    main()
