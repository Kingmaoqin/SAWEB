# sa_data_manager.py (Updated Version)

import pandas as pd
from typing import Optional

class DataManager:
    """A simple singleton-like class to manage the active dataset."""
    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._file_name: Optional[str] = None  # name of the merged/active dataset

        # Optional DataFrames for different modalities. They default to ``None``
        # and are only populated when multimodal data is explicitly loaded.
        self.tabular_df: Optional[pd.DataFrame] = None
        self.image_df: Optional[pd.DataFrame] = None
        self.sensor_df: Optional[pd.DataFrame] = None

    def load_data(self, data: pd.DataFrame, file_name: str): # Add file_name parameter
        """Loads a pandas DataFrame and its file name into the manager."""
        print(f"Data from '{file_name}' loaded into SA Data Manager.")
        self._data = data
        self._file_name = file_name

    def get_data(self) -> Optional[pd.DataFrame]:
        """Retrieves the loaded DataFrame."""
        return self._data

    def load_multimodal_data(
        self,
        *,
        tabular_df: Optional[pd.DataFrame] = None,
        image_df: Optional[pd.DataFrame] = None,
        sensor_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Loads optional DataFrames for different data modalities.

        Parameters
        ----------
        tabular_df, image_df, sensor_df:
            DataFrames containing tabular, image-level or sensor-level
            features respectively. Any of them may be omitted. The first
            non-``None`` DataFrame is also exposed as ``_data`` for backward
            compatibility with existing single-modality code.
        """

        self.tabular_df = tabular_df
        self.image_df = image_df
        self.sensor_df = sensor_df

        # Preserve backward compatibility: expose the first available DataFrame
        # as ``_data`` so that existing calls to ``get_data`` keep working.
        primary = tabular_df or image_df or sensor_df
        if primary is not None:
            self._data = primary
            # ``load_multimodal_data`` may not have a meaningful file name, so
            # we keep the previous one unless a new dataset is provided via
            # ``load_data``.

    def get_data_summary(self) -> dict:
        """Returns a summary of the loaded data."""
        if self._data is None and not any([self.tabular_df, self.image_df, self.sensor_df]):
            return {"error": "No data has been loaded. Please upload a dataset on the 'Run Models' page."}

        summary = {}
        if self._data is not None:
            summary.update(
                {
                    "file_name": self._file_name,
                    "num_rows": int(self._data.shape[0]),
                    "num_columns": int(self._data.shape[1]),
                    "column_names": self._data.columns.tolist(),
                    "missing_values": self._data.isnull().sum().to_dict(),
                }
            )

        modalities = {}
        for name, df in [
            ("tabular", self.tabular_df),
            ("image", self.image_df),
            ("sensor", self.sensor_df),
        ]:
            if df is not None:
                modalities[name] = {
                    "num_rows": int(df.shape[0]),
                    "num_columns": int(df.shape[1]),
                    "missing_values": df.isnull().sum().to_dict(),
                }

        if modalities:
            summary["modalities"] = modalities

        return summary

