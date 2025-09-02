import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Callable


from sa_data_manager import DataManager
from preprocess.image import load_image, default_image_transform

__all__ = [
    "make_intervals",
    "build_supervision",
    "infer_feature_cols",
    "MultiTaskDataset",
    "MultiModalDataset",
    "create_dataloaders",
]

def make_intervals(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    n_bins: int = 30,
    method: str = "quantile",
) -> pd.DataFrame:
    """
    Add 'interval_number' to df using quantile (default) or equal-width binning.
    Ensures event_col is {0,1} int and duration is numeric.
    """
    df = df.copy()
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    if method == "quantile":
        q = pd.qcut(
            df[duration_col].rank(method="first"),
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        df["interval_number"] = (q.astype(int) + 1)
    else:
        q = pd.cut(
            df[duration_col],
            bins=n_bins,
            labels=False,
            duplicates="drop",
        )
        df["interval_number"] = (q.astype(int) + 1)

    # Guarantee int
    df["interval_number"] = df["interval_number"].astype(int)
    return df


def build_supervision(
    intervals: np.ndarray,
    events: np.ndarray,
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one-hot event labels and risk-set masks.
    - labels: one-hot at event bin if event==1, else zeros.
    - masks:  1 up to interval (inclusive), else 0.
    Shapes: (N, T), (N, T)
    """
    N = len(intervals)
    T = int(num_bins)
    labels = np.zeros((N, T), dtype=np.float32)
    masks  = np.zeros((N, T), dtype=np.float32)

    inter = intervals.astype(int)
    ev    = events.astype(int)

    # mask: [0:interval] = 1
    # event one-hot
    for i in range(N):
        k = inter[i]
        if k > 0:
            masks[i, :k] = 1.0
            if ev[i] == 1:
                labels[i, k-1] = 1.0
    return labels, masks


def infer_feature_cols(
    df: pd.DataFrame,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """
    Infer numeric feature columns by excluding obvious non-features and non-numerics.
    """
    if exclude is None:
        exclude = []
    # default excludes
    default_exclude = {"interval_number", "fold", "split", "set"}
    exclude_set = set([c.lower() for c in exclude]) | default_exclude

    num_cols = []
    for c in df.columns:
        cl = c.lower()
        if cl in exclude_set:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    return num_cols


class MultiTaskDataset(Dataset):
    """
    X: (N, P) float32
    y: (N, T) float32 one-hot events
    m: (N, T) float32 risk masks
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, m: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.m = torch.from_numpy(m.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.m[idx]

class MultiModalDataset(Dataset):
    """Dataset handling tabular and image data loaded from :class:`DataManager`.

    Parameters
    ----------
    data_manager: DataManager
        Manager holding the DataFrame with data. The DataFrame must contain
        columns for the tabular features, the survival labels, and optionally
        a column with image file paths.
    feature_cols: List[str]
        Names of numeric columns to be used as tabular features.
    time_col: str
        Column name for the event/censoring time.
    event_col: str
        Column name for the event indicator (1 if event occurred, else 0).
    image_path_col: Optional[str]
        Column containing file paths to the images. If ``None``, image tensors
        are omitted.
    transform: callable, optional
        Transformation applied to images. Defaults to
        :func:`preprocess.image.default_image_transform`.
    encoder: Optional[str]
        Name of a torchvision model to use for feature extraction. Currently
        supports ``"resnet18"``. If ``None``, raw image tensors are returned.
    """

    def __init__(
        self,
        data_manager: DataManager,
        feature_cols: List[str],
        time_col: str,
        event_col: str,
        image_path_col: Optional[str] = None,
        transform: Optional[Callable] = None,
        encoder: Optional[str] = None,
    ):
        df = data_manager.get_data()
        if df is None:
            raise ValueError("DataManager has no data loaded.")

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.time_col = time_col
        self.event_col = event_col
        self.image_path_col = image_path_col
        self.transform = transform or default_image_transform()
        self.encoder = self._init_encoder(encoder)

    def _init_encoder(self, name: Optional[str]):
        if not name:
            return None

        name = name.lower()
        if name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights

            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            modules = list(model.children())[:-1]
            encoder = torch.nn.Sequential(*modules)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            return encoder
        raise ValueError(f"Unknown encoder: {name}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        tabular = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)
        duration = torch.tensor(row[self.time_col], dtype=torch.float32)
        event = torch.tensor(row[self.event_col], dtype=torch.float32)

        image = None
        if self.image_path_col is not None:
            path = row[self.image_path_col]
            image = load_image(path, self.transform)
            if self.encoder is not None:
                with torch.no_grad():
                    image = self.encoder(image.unsqueeze(0)).squeeze()

        return tabular, image, duration, event



def create_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray, m_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,   m_val: np.ndarray,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = MultiTaskDataset(X_train, y_train, m_train)
    val_ds   = MultiTaskDataset(X_val,   y_val,   m_val)

    # make pin/persistent safe for both CPU and GPU + num_workers=0
    pin_memory = torch.cuda.is_available()
    use_persistent = bool(num_workers and num_workers > 0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=use_persistent
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=use_persistent
    )
    return train_loader, val_loader
