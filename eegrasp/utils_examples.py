r"""
Utils Examples
=====

Utils functions used in the examples.
"""

import os 
import json 
import numpy as np

ASSETS_GRAPH_LEARNING = [
    {
        "filename": "data_set_IVa_aa.mat",
        "url": "https://www.bbci.de/competition/download/competition_iii/berlin/100Hz/data_set_IVa_aa_mat.zip"
    }
]

ASSETS = {
    "graph_learning": ASSETS_GRAPH_LEARNING
}


def fetch_data(output_dir, database="graph_learning"):
    """ 
    Fetch data from the internet and save it in the output_dir.

    Parameters
    ----------
    output_dir : str
        Directory where the data will be saved.
    database : str, optional
        Database to fetch data from. Options are: "metro".
    """ 
    for asset_dict in ASSETS[database]:
        filename = asset_dict["filename"]
        url = asset_dict["url"]
        assets_filepath = os.path.join(output_dir, filename)
        if not os.path.isfile(assets_filepath):
            import requests
            import zipfile
            import io
            print(f'Downloading data file to:\n {assets_filepath}')
            r = requests.get(url)
            if url.endswith('.zip'):
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(output_dir)
            else:
                with open(assets_filepath, 'wb') as f:
                    f.write(r.content)