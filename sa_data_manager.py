# sa_data_manager.py (Updated Version)

import pandas as pd
from typing import Optional

class DataManager:
    """A simple singleton-like class to manage the active dataset."""
    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._file_name: Optional[str] = None # Add file_name attribute

    def load_data(self, data: pd.DataFrame, file_name: str): # Add file_name parameter
        """Loads a pandas DataFrame and its file name into the manager."""
        print(f"Data from '{file_name}' loaded into SA Data Manager.")
        self._data = data
        self._file_name = file_name

    def get_data(self) -> Optional[pd.DataFrame]:
        """Retrieves the loaded DataFrame."""
        return self._data

    def get_data_summary(self) -> dict:
        """Returns a summary of the loaded data."""
        if self._data is None:
            return {"error": "No data has been loaded. Please upload a dataset on the 'Run Models' page."}
        
        return {
            "file_name": self._file_name,
            "num_rows": int(self._data.shape[0]),
            "num_columns": int(self._data.shape[1]),
            "column_names": self._data.columns.tolist(),
            "missing_values": self._data.isnull().sum().to_dict()
        }

