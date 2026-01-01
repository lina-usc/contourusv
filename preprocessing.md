# Background
For our USV analysis pipeline we have two pipelines:
- USV Detection Pipeline
- USV Clustering Pipeline

Both are working pipelines but requires improvement to perform better than state-of-the-art.

## Improvement
- In both cases the data preprocessing stage needs to be revisited
- For enhancements because with cleaner data we will get performance improvements automatically. 

Below I will go into detail how each pipeline works, where and how the data preprocessing is done, and what kind of improvements are required.

# Detection Pipeline (ContourUSV)
In the detection pipeline we have 7 scripts in total.

- `generate_annotation.py`: 
    - convert manual annotations (which may be html/csv files based on the dataset) to csv files with specific (required) columns:
        - file_name, begin_time, end_time, low_freq, high_freq, and usv_type (label: 22 or 50 kHz) 
    - The annotations serve as the gold standard for the final step (evaluation) in the `evaluation_of_all_experiments.py` script

- `main.py`: 
    - loads audio files, 
    - calls `gen_figures` function to run detection and save figures

- `spectrogram.py`: 
    - contains `gen_spec` function
    - called in `main.py` to generate spectrograms from audio files
    - returns spectrogram data

- `preprocessing.py`: 
    - contains `clean_spec` function
    - called in `main.py` to preprocess and clean the generated spectrograms
    - returns closing image after morphological operations

- `detection.py`: 
    - contains `detect_contours` function
    - called in `main.py` to detect contours in the clean spectrograms
    - returns detected spectrogram and annotations

-  `evaluation_of_all_experiments.py`:
    - using predicted labels from the annotations generated in the detection, evaluation is performed
    - mean precion, recall, f1 score, and specificity is calculated

- `paired_ttest.py`:
    - for performing paired ttest on generated results

Noise reduction techniques need to implemented in spectrogram generation and preprocessing steps (`spectrogram.py` and `preprocessing.py`) to get a cleaner spectrogram without losing USV information.

# Data preprocessing in clustering: 
The clustering part is still in its initial stage. And it is very important to get the spectrogram preprocessing right in this stage to get good results with any clustering algorithm. Right now, after detecting the USVs in the spectrograms, I am using those detection coordinates to do precise localization (zoom in) of the USV calls and use those zoomed in spectrograms to feed into the clustering algorithms. Since, our detector also detects some noise as USVs in the spectrograms, the data for clustering also needs to be processed well to filter out those wrong detections.