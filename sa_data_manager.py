# sa_data_manager.py (Updated Version)

import io
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from PIL import Image
from typing import Any, Optional


class DataManager:
    """A simple singleton-like class to manage the active dataset."""
    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._sensor_data: Optional[Any] = None
        self._image_data: Optional[Any] = None
        self._file_name: Optional[str] = None  # Add file_name attribute

    def load_data(self, data: pd.DataFrame, file_name: str): # Add file_name parameter
        """Loads a pandas DataFrame and its file name into the manager."""
        print(f"Data from '{file_name}' loaded into SA Data Manager.")
        self._data = data
        self._file_name = file_name

    def load_sensor_data(self, data: Any):
        """Loads sensor data into the manager."""
        print("Sensor data loaded into SA Data Manager.")
        self._sensor_data = data

    def load_image_data(self, data: Any):
        """Loads image data into the manager."""
        print("Image data loaded into SA Data Manager.")
        self._image_data = data

    def get_data(self) -> Optional[pd.DataFrame]:
        """Retrieves the loaded DataFrame."""
        return self._data

    def get_sensor_data(self) -> Optional[Any]:
        """Retrieves the loaded sensor data."""
        return self._sensor_data

    def get_image_data(self) -> Optional[Any]:
        """Retrieves the loaded image data."""
        return self._image_data

    def get_data_summary(self) -> dict:
        """Returns a summary of the loaded data."""

        if self._data is None:
            return {"error": "No data has been loaded. Please upload a dataset on the 'Run Models' page."}

        return {

            "file_name": self._file_name,
            "num_rows": int(self._data.shape[0]) if self._data is not None else None,
            "num_columns": int(self._data.shape[1]) if self._data is not None else None,
            "column_names": self._data.columns.tolist() if self._data is not None else None,
            "missing_values": self._data.isnull().sum().to_dict() if self._data is not None else None,
            "modalities": {}
        }

    def get_sensor_summary(self) -> Union[List[Dict[str, int]], Dict[str, str]]:
        """Return length and channel information for sensor-like columns."""
        if self._data is None:
            return {"error": "No data has been loaded. Please upload a dataset on the 'Run Models' page."}

        summaries: List[Dict[str, int]] = []
        for col in self._data.columns:
            series = self._data[col].dropna()
            if series.empty:
                continue
            first = series.iloc[0]
            if isinstance(first, (list, tuple, np.ndarray, pd.Series)):
                arr = np.asarray(first)
                length = int(arr.shape[0])
                channels = int(arr.shape[1]) if arr.ndim > 1 else 1
                summaries.append({"column": col, "length": length, "channels": channels})
        if not summaries:
            return {"error": "No sensor data columns detected."}
        return summaries

    def get_image_summary(self) -> Union[List[Dict[str, Any]], Dict[str, str]]:
        """Return thumbnails and shapes for image-like columns."""
        if self._data is None:
            return {"error": "No data has been loaded. Please upload a dataset on the 'Run Models' page."}

        summaries: List[Dict[str, Any]] = []
        for col in self._data.columns:
            series = self._data[col].dropna()
            if series.empty:
                continue
            first = series.iloc[0]
            image = None
            if isinstance(first, Image.Image):
                image = first
            elif isinstance(first, (bytes, bytearray)):
                try:
                    image = Image.open(io.BytesIO(first))
                except Exception:
                    image = None
            elif isinstance(first, np.ndarray):
                try:
                    image = Image.fromarray(first)
                except Exception:
                    image = None
            elif isinstance(first, str) and first.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
                try:
                    image = Image.open(first)
                except Exception:
                    image = None

            if image is not None:
                shape = (image.height, image.width, len(image.getbands()))
                summaries.append({"column": col, "image": image, "shape": shape})

        if not summaries:
            return {"error": "No image data columns detected."}
        return summaries

