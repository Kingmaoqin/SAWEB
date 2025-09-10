import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List, Tuple, Any

__all__ = ["MultiModalDataset", "multimodal_collate_fn", "create_multimodal_dataloaders"]


class MultiModalDataset(Dataset):
    """Dataset handling multiple modalities stored in separate DataFrames.

    Each modality DataFrame must contain an identifier column (``id_col``)
    used to align rows across the modalities.  Missing modalities for a given
    sample are represented by a zero-vector and a corresponding mask of ``0``.

    Parameters
    ----------
    image_df, sensor_df, table_df:
        DataFrames containing features for each modality.  Any of them can be
        ``None`` if the modality is not available.
    id_col:
        Column name that uniquely identifies a sample across all DataFrames.
    fill_value:
        Value used to pad missing numeric modalities.  Defaults to ``0.0``.
    """

    def __init__(
        self,
        image_df: Optional[pd.DataFrame] = None,
        sensor_df: Optional[pd.DataFrame] = None,
        table_df: Optional[pd.DataFrame] = None,
        id_col: str = "id",
        fill_value: float = 0.0,
    ) -> None:
        self.id_col = id_col
        self.fill_value = fill_value

        self.modalities: Dict[str, pd.DataFrame] = {}
        self.feature_dims: Dict[str, int] = {}

        for name, df in [
            ("image", image_df),
            ("sensor", sensor_df),
            ("table", table_df),
        ]:
            if df is not None:
                if id_col not in df.columns:
                    raise ValueError(f"{name}_df must contain column '{id_col}'")
                df = df.set_index(id_col)
                self.modalities[name] = df
                # number of feature columns (all except id)
                self.feature_dims[name] = df.shape[1]

        if not self.modalities:
            raise ValueError("At least one modality DataFrame must be provided")

        # Unified index across all modalities
        id_sets = [set(df.index) for df in self.modalities.values()]
        self.ids: List[Any] = sorted(set.union(*id_sets))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        sample_id = self.ids[idx]
        item: Dict[str, Any] = {"id": sample_id}

        for name, df in self.modalities.items():
            if sample_id in df.index:
                row = df.loc[sample_id]
                # Always convert numeric data to float tensors
                values = row.to_numpy()
                tensor = torch.as_tensor(values, dtype=torch.float32)
                mask = torch.tensor(1.0, dtype=torch.float32)
            else:
                tensor = torch.full(
                    (self.feature_dims[name],),
                    self.fill_value,
                    dtype=torch.float32,
                )
                mask = torch.tensor(0.0, dtype=torch.float32)
            item[name] = tensor
            item[f"{name}_mask"] = mask

        return item


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function to stack multimodal batches.

    Handles dictionaries returned by :class:`MultiModalDataset` by stacking
    tensor values and grouping non-tensor values into lists.
    """
    collated: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values
    return collated


def create_multimodal_dataloaders(
    image_train: Optional[pd.DataFrame] = None,
    sensor_train: Optional[pd.DataFrame] = None,
    table_train: Optional[pd.DataFrame] = None,
    image_val: Optional[pd.DataFrame] = None,
    sensor_val: Optional[pd.DataFrame] = None,
    table_val: Optional[pd.DataFrame] = None,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders for multi-modal data."""
    train_ds = MultiModalDataset(
        image_df=image_train,
        sensor_df=sensor_train,
        table_df=table_train,
    )
    val_ds = MultiModalDataset(
        image_df=image_val,
        sensor_df=sensor_val,
        table_df=table_val,
    )

    pin_memory = torch.cuda.is_available()
    use_persistent = bool(num_workers and num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        collate_fn=multimodal_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        collate_fn=multimodal_collate_fn,
    )
    return train_loader, val_loader
