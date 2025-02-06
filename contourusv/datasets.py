# Add code to fetch datasets/testing data

import urllib.request
from pathlib import Path

def get_usvseg(file_stem=None):
    """
    file can be "rat_distressed", ...
    """
    # TODO: Document
    # TODO: Download only if not already available...
    # TODO: Accept specified root dataset path...

    if file_stem is None:
        # dowload all files
        # TODO
    else:
        url = f"https://zenodo.org/records/3428024/files/{file_stem}.zip?download=1"
        dataset_path = (Path(__file__).parent / "dataset" / "usvseg")
        dataset_path.mkdir(parents=True, exist_ok=True)

        urllib.request.urlretrieve(url, dataset_path / f"filestem.zip")
        # TODO: unzip...

