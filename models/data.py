import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

__all__ = ["make_intervals", "build_supervision", "infer_feature_cols",
           "MultiTaskDataset", "create_dataloaders"]

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


def create_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray, m_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,   m_val: np.ndarray,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = MultiTaskDataset(X_train, y_train, m_train)
    val_ds   = MultiTaskDataset(X_val,   y_val,   m_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    return train_loader, val_loader
