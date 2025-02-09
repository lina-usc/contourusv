import pandas as pd
from pathlib import Path

def load_html_usv(file_name):
    """
    Load USV annotations from DeepSqueak HTML file.

    Parameters
    ----------
    file_name : Path
        Path to DeepSqueak HTML annotation file

    Returns
    -------
    DataFrame
        Processed annotations with columns:
        ['begin_time', 'end_time', 'usv_type']
    """
    metadata, data = pd.read_html(file_name)
    for df in [metadata, data]:
        df.rename(columns=df.iloc[0], inplace=True)
        df.drop(df.index[0], inplace=True)

    num_columns = {
        'Start Time (s)': 'begin_time',
        'Stop Time (s)': 'end_time',
        'Pattern Label': 'usv_type'
    }
    for col in ['Start Time (s)', 'Stop Time (s)']:
        data[col] = data[col].astype(float)
    data.rename(columns=num_columns, inplace=True)

    return data


def load_excel_usv(file_name):
    """
    Load USV annotations from Excel file.

    Parameters
    ----------
    file_name : Path
        Path to Excel annotation file (.xlsx)

    Returns
    -------
    DataFrame
        Processed annotations with columns:
        ['begin_time', 'end_time', 'low_freq', 'high_freq', 'usv_type']
    """
    data = pd.read_excel(file_name, engine='openpyxl')
    num_columns = {
        'Begin Time (s)': 'begin_time',
        'End Time (s)': 'end_time',
        'Low Freq (kHz)': 'low_freq',
        'High Freq (kHz)': 'high_freq',
        'Label': 'usv_type'
    }
    for col in ['Begin Time (s)', 'End Time (s)', 'Low Freq (kHz)', 'High Freq (kHz)']:
        data[col] = data[col].astype(float)
    data.rename(columns=num_columns, inplace=True)
    return data


def load_csv_usv(file_name):
    """
    Load simple timestamp annotations from CSV file.

    Parameters
    ----------
    file_name : Path
        Path to CSV annotation file

    Returns
    -------
    DataFrame
        Processed annotations with columns:
        ['begin_time', 'end_time']
    """
    data = pd.read_csv(file_name, header=None, names=['begin_time', 'end_time'], usecols=[0, 1])
    data['begin_time'] = data['begin_time'].astype(float)
    data['end_time'] = data['end_time'].astype(float)
    return data


def save_annotations(files, audio_file_name, output_path, file_ext,
                     freq_min=18000, freq_max=30000):
    """
    Convert annotation files to standardized format and save.

    Parameters
    ----------
    files : list
        List of annotation file paths
    audio_file_name : Path
        Corresponding audio file path
    output_path : Path
        Directory to save processed annotations
    file_ext : str
        Annotation file type (.html, .xlsx, .csv)
    freq_min : int, optional
        Default minimum frequency (Hz) for CSV/HTML (default: 18000)
    freq_max : int, optional
        Default maximum frequency (Hz) for CSV/HTML (default: 30000)
    """
    annotations = []
      loaders = {'.html': load_html_usv, '.csv': load_csv_usv, '.xlsx': load_excel_usv}
      for f in files:
        usv_data = loaders[file_ext](f)
        low_freq = freq_min
        high_freq = freq_max        
        for _, row in usv_data.iterrows():
            if file_ext == '.html':
                usv_type = row['usv_type']
            elif file_ext == '.xlsx':
                low_freq = row.get('low_freq', freq_min) * 1000
                high_freq = row.get('high_freq', freq_max) * 1000
                usv_type =  row.get('usv_type', '')
            elif file_ext == '.csv':
                 usv_type = ''
            
            annotations.append({
                'begin_file': audio_file_name.stem,
                'begin_time': row['begin_time'],
                'end_time': row['end_time'],
                'low_freq': low_freq,
                'high_freq': high_freq,
                'usv_type': usv_type
                })

    if annotations:
        annotations_df = pd.DataFrame(annotations)
        output_file_name = output_path / (Path(audio_file_name).stem + '.csv')
        annotations_df.to_csv(output_file_name, sep='\t', index=False)
        print(f"Saved annotations for {audio_file_name} to {output_file_name}")


def generate_annotations(experiment, trial, root_path, file_ext):
    """
    Generate ground truth annotations for all audio files in experiment.

    Parameters
    ----------
    experiment : str
        Experiment name
    trial : str
        Trial/condition name
    root_path : Path
        Root data directory
    file_ext : str
        Annotation file extension (.html, .xlsx, .csv)

    Raises
    ------
    ValueError
        If invalid file extension is provided
    """
    print(f"Processing {experiment} {trial} experiment...")
    files_path = Path(root_path) / experiment / trial
    output_path = Path(f'{root_path}/output/{experiment}/{trial}/ground_truth_annotations')
    output_path.mkdir(parents=True, exist_ok=True)

    if file_ext == '.html':
        html_files = sorted(list(files_path.rglob("*.html")))
    elif file_ext == '.xlsx':
        xlsx_files = sorted(list(files_path.rglob("*.xlsx")))
    elif file_ext == '.csv':
        csv_files = sorted(list(files_path.rglob("*.csv")))
    else:
        raise ValueError("Invalid file extension. Please use html, xlsx, or csv.")

    audio_files = sorted(list(files_path.rglob("*.wav")) + list(files_path.rglob("*.WAV")))

    for audio_file in audio_files:
        # Match files
        stem_parts = audio_file.stem.split('_')[0:3]

        if file_ext == '.html':
            matched_html = [f for f in html_files if f.stem.split('_')[0:3] == stem_parts]
            if matched_html:
                save_annotations(matched_html, audio_file, output_path, file_ext)
        elif file_ext == '.xlsx':
            matched_xlsx = [f for f in xlsx_files if f.stem.split('_')[0:3] == stem_parts]
            if matched_xlsx:
                save_annotations(matched_xlsx, audio_file, output_path, file_ext)
        elif file_ext == '.csv':
            matched_csv = [f for f in csv_files if f.stem.split('_')[0:3] == stem_parts]
            if matched_csv:
                save_annotations(matched_csv, audio_file, output_path, file_ext)
        else:
            raise ValueError("Invalid file extension. Please use html, xlsx, or csv.")