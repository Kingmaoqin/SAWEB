import os
import tempfile
import zipfile
from io import BytesIO
from typing import Dict

import pandas as pd
import requests
import streamlit as st

from sa_data_manager import DataManager

# Built-in datasets with description and source link
DATASETS: Dict[str, Dict[str, str]] = {
    "Lung Cancer": {
        "url": "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/lung.csv",
        "description": "Survival in patients with advanced lung cancer.",
        "link": "https://vincentarelbundock.github.io/Rdatasets/doc/survival/lung.html",
    },
    "Veteran": {
        "url": "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/veteran.csv",
        "description": "Veteran's Administration lung cancer trial data.",
        "link": "https://vincentarelbundock.github.io/Rdatasets/doc/survival/veteran.html",
    },
}

def _download_dataset(url: str, tmpdir: str) -> pd.DataFrame:
    """Download a dataset. If zipped, extract to tmpdir and return DataFrame."""
    resp = requests.get(url)
    resp.raise_for_status()

    filename = os.path.basename(url)
    filepath = os.path.join(tmpdir, filename)
    with open(filepath, "wb") as f:
        f.write(resp.content)

    # Handle zip files
    if filename.endswith(".zip"):
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            zf.extractall(tmpdir)
            # Assume first CSV inside zip is target
            for member in zf.namelist():
                if member.lower().endswith(".csv"):
                    return pd.read_csv(os.path.join(tmpdir, member))
        raise ValueError("No CSV file found in the zip archive.")

    # Non-zip: assume CSV
    return pd.read_csv(filepath)

def show():
    st.title("Model Training")

    dataset_name = st.selectbox("Select a built-in dataset", list(DATASETS.keys()))
    info = DATASETS[dataset_name]
    st.write(info["description"])
    st.markdown(f"[Learn more about this dataset]({info['link']})")

    if st.button("Load Dataset"):
        with st.spinner("Downloading dataset..."):
            tmpdir = tempfile.mkdtemp()
            df = _download_dataset(info["url"], tmpdir)
            data_manager: DataManager = st.session_state.get("data_manager", DataManager())
            data_manager.load_data(df, f"{dataset_name}.csv")
            st.session_state.data_manager = data_manager
            st.success(f"'{dataset_name}' loaded into DataManager.")
            st.json(data_manager.get_data_summary())
