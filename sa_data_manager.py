# sa_data_manager.py (Updated Version)

import pandas as pd
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
        if self._data is None and self._sensor_data is None and self._image_data is None:
            return {
                "error": "No data has been loaded. Please upload a dataset on the 'Run Models' page."
            }

        summary = {
            "file_name": self._file_name,
            "num_rows": int(self._data.shape[0]) if self._data is not None else None,
            "num_columns": int(self._data.shape[1]) if self._data is not None else None,
            "column_names": self._data.columns.tolist() if self._data is not None else None,
            "missing_values": self._data.isnull().sum().to_dict() if self._data is not None else None,
            "modalities": {}
        }

        if self._data is not None:
            summary["modalities"]["tabular"] = {"shape": tuple(self._data.shape)}

        if self._sensor_data is not None:
            shape = getattr(self._sensor_data, "shape", None)
            summary["modalities"]["sensor"] = {"shape": tuple(shape) if shape is not None else None}

        if self._image_data is not None:
            shape = getattr(self._image_data, "shape", None)
            summary["modalities"]["image"] = {"shape": tuple(shape) if shape is not None else None}

        return summary

