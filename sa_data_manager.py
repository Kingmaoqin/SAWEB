# sa_data_manager.py (Updated Version)

import pandas as pd
import zipfile
from typing import Optional, List

class DataManager:
    """A simple singleton-like class to manage the active dataset."""
    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._file_name: Optional[str] = None # Add file_name attribute
        self._image_files: List[str] = []

    def load_data(self, data: pd.DataFrame, file_name: str): # Add file_name parameter
        """Loads a pandas DataFrame and its file name into the manager."""
        print(f"Data from '{file_name}' loaded into SA Data Manager.")
        self._data = data
        self._file_name = file_name

    def load_sensor(self, file) -> List[str]:
        """Loads sensor data from a CSV or JSON file."""
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith('.json'):
            data = pd.read_json(file)
        else:
            raise ValueError("Unsupported sensor file format.")
        self.load_data(data, file.name)
        return [file.name]

    def load_image(self, file) -> List[str]:
        """Loads image file names from a ZIP archive."""
        with zipfile.ZipFile(file) as zf:
            image_names = [
                info.filename
                for info in zf.infolist()
                if info.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            ]
        self._image_files = image_names
        self._file_name = file.name
        return image_names

    def get_image_files(self) -> List[str]:
        """Returns the names of loaded image files."""
        return self._image_files

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

