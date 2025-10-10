# sa_data_manager.py

import pandas as pd
from typing import Optional

class DataManager:
    """A simple singleton-like class to manage the active dataset."""
    def __init__(self):
        self._data: Optional[pd.DataFrame] = None

    def load_data(self, data: pd.DataFrame):
        """Loads a pandas DataFrame into the manager."""
        print("Data loaded into SA Data Manager.")
        self._data = data

    def get_data(self) -> Optional[pd.DataFrame]:
        """Retrieves the loaded DataFrame."""
        if self._data is None:
            print("Warning: Attempted to access data, but none is loaded.")
        return self._data

    def get_data_summary(self) -> dict:
        """Returns a summary of the loaded data."""
        if self._data is None:
            return {"error": "No data has been loaded. Please upload a dataset on the 'Run Models' page."}
        
        return {
            "num_rows": int(self._data.shape[0]),
            "num_columns": int(self._data.shape[1]),
            "column_names": self._data.columns.tolist(),
            "missing_values": self._data.isnull().sum().to_dict()
        }

# Global instance to be imported by other modules
sa_data_manager = DataManager()